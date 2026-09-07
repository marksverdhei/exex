import torch

from exex.trainer import ExpertTrainer


def _masked_batch(ids_a, ids_b, pad_id=0):
    """Right-pad two sequences the way train_expert.py does after the fix."""
    n = max(len(ids_a), len(ids_b))
    input_ids = torch.full((2, n), pad_id)
    attention_mask = torch.zeros((2, n), dtype=torch.long)
    for row, ids in enumerate((ids_a, ids_b)):
        input_ids[row, : len(ids)] = torch.tensor(ids)
        attention_mask[row, : len(ids)] = 1
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return input_ids, attention_mask, labels


class TestPaddingInvariance:
    def test_loss_ignores_pad_positions(self, tiny_gemma4_moe):
        """Task loss with masked padding must equal the unpadded per-sequence losses."""
        model = tiny_gemma4_moe.eval()
        trainer = ExpertTrainer(model, [1])
        a = torch.randint(1, 256, (24,)).tolist()
        b = torch.randint(1, 256, (9,)).tolist()
        input_ids, attention_mask, labels = _masked_batch(a, b)
        with torch.no_grad():
            padded, _ = trainer.compute_loss(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            # reference: token-weighted mean of the two unpadded sequence losses
            tot, n = 0.0, 0
            for ids in (a, b):
                t = torch.tensor([ids])
                loss, _ = trainer.compute_loss(input_ids=t, labels=t)
                tot += loss.item() * (len(ids) - 1)
                n += len(ids) - 1
        assert abs(padded.item() - tot / n) < 1e-3

    def test_naive_labels_differ(self, tiny_gemma4_moe):
        """Guard: unmasked labels (the pre-fix behaviour) give a different loss."""
        model = tiny_gemma4_moe.eval()
        trainer = ExpertTrainer(model, [1])
        a = torch.randint(1, 256, (24,)).tolist()
        b = torch.randint(1, 256, (9,)).tolist()
        input_ids, attention_mask, labels = _masked_batch(a, b)
        with torch.no_grad():
            masked, _ = trainer.compute_loss(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            naive, _ = trainer.compute_loss(input_ids=input_ids, labels=input_ids.clone())
        assert abs(masked.item() - naive.item()) > 1e-4
