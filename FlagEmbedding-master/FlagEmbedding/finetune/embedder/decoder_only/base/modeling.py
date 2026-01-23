import logging
from transformers import AutoModel, PreTrainedModel, PreTrainedTokenizer
from FlagEmbedding.abc.finetune.embedder import AbsEmbedderModel

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributed as dist
from transformers.file_utils import ModelOutput

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Union

logger = logging.getLogger(__name__)

@dataclass
class EmbedderOutput(ModelOutput):
    """
    Output information returned by the model.
    """
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None

class BiDecoderOnlyEmbedderModel(AbsEmbedderModel):
    """Embedder model class for decoder only model.

    Args:
        base_model (PreTrainedModel): The base model to train on.
        tokenizer (PreTrainedTokenizer, optional): The tokenizer to use. Defaults to ``None``.
        negatives_cross_device (bool, optional): If True, will compute cross devices negative loss. Defaults to ``False``.
        temperature (float, optional): Temperature to control the scale of scores. Defaults to ``1.0``.
        sub_batch_size (int, optional): Sub-batch size during encoding. If negative, will not split to sub-batch.
            Defaults to ``-1``.
        kd_loss_type (str, optional): Type of knowledge distillation loss. Defaults to ``'kl_div'``.
        sentence_pooling_method (str, optional): Pooling method to get sentence embedding. Defaults to ``'last_token'``.
        normalize_embeddings (bool, optional): If True, normalize the embedding vector. Defaults to ``False``.
    """
    TRANSFORMER_CLS = AutoModel

    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer = None,
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        sub_batch_size: int = -1,
        kd_loss_type: str = 'kl_div',
        sentence_pooling_method: str = 'last_token',
        normalize_embeddings: bool = False,
    ):
        super().__init__(
            base_model,
            tokenizer=tokenizer,
            negatives_cross_device=negatives_cross_device,
            temperature=temperature,
            sub_batch_size=sub_batch_size,
            kd_loss_type=kd_loss_type,
        )
        self.sentence_pooling_method = sentence_pooling_method
        self.normalize_embeddings = normalize_embeddings
        ###
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.cross_entropy_none = torch.nn.CrossEntropyLoss(reduction="none")

    def encode(self, features):
        """
        Encode and get the embedding.

        Args:
            features (Union[list, dict]): Features feed to the model.

        Returns:
            torch.Tensor: The embedding vectors.
        """
        if features is None:
            return None
        if not isinstance(features, list):
            if self.sub_batch_size is not None and self.sub_batch_size > 0:
                all_p_reps = []
                for i in range(0, len(features['attention_mask']), self.sub_batch_size):
                    end_inx = min(i + self.sub_batch_size, len(features['attention_mask']))
                    sub_features = {}
                    for k, v in features.items():
                        sub_features[k] = v[i:end_inx]
                    last_hidden_state = self.model(**sub_features, return_dict=True).last_hidden_state
                    p_reps = self._sentence_embedding(last_hidden_state, sub_features['attention_mask'])
                    all_p_reps.append(p_reps)
                all_p_reps = torch.cat(all_p_reps, 0).contiguous()
                if self.normalize_embeddings:
                    all_p_reps = torch.nn.functional.normalize(all_p_reps, dim=-1)
                return all_p_reps.contiguous()
            else:
                last_hidden_state = self.model(**features, return_dict=True).last_hidden_state
                all_p_reps = self._sentence_embedding(last_hidden_state, features['attention_mask'])
                if self.normalize_embeddings:
                    all_p_reps = torch.nn.functional.normalize(all_p_reps, dim=-1)
                return all_p_reps.contiguous()
        else:
            all_p_reps = []
            for sub_features in features:
                last_hidden_state = self.model(**sub_features, return_dict=True).last_hidden_state
                p_reps = self._sentence_embedding(last_hidden_state, sub_features['attention_mask'])
                all_p_reps.append(p_reps)
            all_p_reps = torch.cat(all_p_reps, 0).contiguous()
            if self.normalize_embeddings:
                all_p_reps = torch.nn.functional.normalize(all_p_reps, dim=-1)
            return all_p_reps.contiguous()

    def _sentence_embedding(self, last_hidden_state, attention_mask):
        """Use the pooling method to get the sentence embedding.

        Args:
            last_hidden_state (torch.Tensor): The model output's last hidden state.
            attention_mask (torch.Tensor): Mask out padding tokens during pooling.

        Raises:
            NotImplementedError: Specified pooling method not implemented.

        Returns:
            torch.Tensor: The sentence embeddings.
        """
        if self.sentence_pooling_method == "cls":
            return last_hidden_state[:, 0]
        elif self.sentence_pooling_method == "mean":
            s = torch.sum(
                last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1
            )
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == "last_token":
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                return last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                return last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device),
                    sequence_lengths,
                ]
        else:
            raise NotImplementedError(f"pooling method {self.sentence_pooling_method} not implemented")

    def compute_score(self, q_reps, p_reps):
        """Computes the scores between query and passage representations.

        Args:
            q_reps (torch.Tensor): Query representations.
            p_reps (torch.Tensor): Passage representations.

        Returns:
            torch.Tensor: The computed scores, adjusted by temperature.
        """
        scores = self._compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def _compute_similarity(self, q_reps, p_reps):
        """Computes the similarity between query and passage representations using inner product.

        Args:
            q_reps (torch.Tensor): Query representations.
            p_reps (torch.Tensor): Passage representations.

        Returns:
            torch.Tensor: The computed similarity matrix.
        """
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    # def compute_loss(self, scores, target):
    #     """Compute the loss using cross entropy.

    #     Args:
    #         scores (torch.Tensor): Computed score.
    #         target (torch.Tensor): The target value.

    #     Returns:
    #         torch.Tensor: The computed cross entropy loss.
    #     """
    #     return self.cross_entropy(scores, target)

    def compute_loss(self, scores, target, reweight_rates=None):
        if reweight_rates is None:
            return self.cross_entropy(scores, target)   # 标量
    
        if not torch.is_tensor(reweight_rates):
            w = torch.tensor(reweight_rates, device=scores.device, dtype=scores.dtype)
        else:
            w = reweight_rates.to(device=scores.device, dtype=scores.dtype)
        w = w.view(-1)
    
        per_sample = self.cross_entropy_none(scores, target)  # (B,)
        return (per_sample * w).sum() / w.sum().clamp_min(1e-12)

    def gradient_checkpointing_enable(self, **kwargs):
        """
        Activates gradient checkpointing for the current model.
        """
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        """
        Enables the gradients for the input embeddings.
        """
        self.model.enable_input_require_grads(**kwargs)

    def _compute_in_batch_neg_loss(self, q_reps, p_reps, teacher_targets=None, reweight_rates=None, compute_score_func=None, **kwargs):
        """
        Compute loss when only using in-batch negatives
        """
        group_size = p_reps.size(0) // q_reps.size(0)

        if compute_score_func is None:
            scores = self.compute_score(q_reps, p_reps) # (batch_size, batch_size * group_size)
        else:
            scores = compute_score_func(q_reps, p_reps, **kwargs)   # (batch_size, batch_size * group_size)
        
        if teacher_targets is not None:
            # compute kd loss
            if self.kd_loss_type == "kl_div":
                student_scores = self.get_local_score(q_reps, p_reps, scores) # (batch_size, group_size)

                loss = self.distill_loss(self.kd_loss_type, teacher_targets, student_scores, group_size)

                idxs = torch.arange(q_reps.size(0), device=q_reps.device, dtype=torch.long)
                targets = idxs * (p_reps.size(0) // q_reps.size(0)) # (batch_size)
                loss += self.compute_loss(scores, targets)
            elif self.kd_loss_type == "m3_kd_loss":
                loss = self.distill_loss(self.kd_loss_type, teacher_targets, scores, group_size)
            else:
                raise ValueError(f"Invalid kd_loss_type: {self.kd_loss_type}")
        else:
            idxs = torch.arange(q_reps.size(0), device=q_reps.device, dtype=torch.long)
            targets = idxs * group_size # (batch_size)
            loss = self.compute_loss(scores, targets, reweight_rates=reweight_rates)

        return scores, loss

    
    def forward(
        self, 
        queries: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None, 
        passages: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        teacher_scores: Union[None, List[float]] = None,
        reweight_rates: List[float] = None,
        no_in_batch_neg_flag: bool = False,
    ):
        """The computation performed at every call.

        Args:
            queries (Union[Dict[str, Tensor], List[Dict[str, Tensor]]], optional): Input queries. Defaults to ``None``.
            passages (Union[Dict[str, Tensor], List[Dict[str, Tensor]]], optional): Input passages. Defaults to ``None``.
            teacher_scores (Union[None, List[float]], optional): Teacher scores for distillation. Defaults to ``None``.
            no_in_batch_neg_flag (bool, optional): If True, use no in-batch negatives and no cross-device negatives. Defaults to ``False``.

        Returns:
            EmbedderOutput: Output of the forward call of model.
        """
        q_reps = self.encode(queries) # (batch_size, dim)
        p_reps = self.encode(passages) # (batch_size * group_size, dim)

        if self.training:
            if teacher_scores is not None:
                teacher_scores = torch.tensor(teacher_scores, device=q_reps.device)
                teacher_scores = teacher_scores.view(q_reps.size(0), -1).detach()   # (batch_size, group_size)
                teacher_targets = F.softmax(teacher_scores, dim=-1)  # (batch_size, group_size)
            else:
                teacher_targets = None

            if no_in_batch_neg_flag:
                compute_loss_func = self._compute_no_in_batch_neg_loss
            else:
                if self.negatives_cross_device:
                    compute_loss_func = self._compute_cross_device_neg_loss
                else:
                    compute_loss_func = self._compute_in_batch_neg_loss

            scores, loss = compute_loss_func(q_reps, p_reps, teacher_targets=teacher_targets, reweight_rates=reweight_rates)
        else:
            loss = None

        return EmbedderOutput(
            loss=loss,
        )

    def save(self, output_dir: str):
        """Save the model to the directory.

        Args:
            output_dir (str): Directory for saving the model.
        """
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)
