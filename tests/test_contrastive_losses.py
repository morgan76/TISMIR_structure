import torch

from tismir.losses import audio_audio_supervised_contrastive, text_to_audio_infonce


def test_text_to_audio_infonce_is_lower_for_matching_tokens():
    audio_tokens = torch.tensor([[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]])
    text_tokens = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([[0, 0, 1, 1]])

    good_loss = text_to_audio_infonce(audio_tokens, text_tokens, targets, temperature=0.1)
    bad_loss = text_to_audio_infonce(audio_tokens.flip(dims=[1]), text_tokens, targets, temperature=0.1)

    assert good_loss < bad_loss


def test_audio_audio_supervised_contrastive_handles_labels_without_repeats():
    audio_tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    targets = torch.tensor([[0, 1]])

    loss = audio_audio_supervised_contrastive(audio_tokens, targets, temperature=0.1)

    assert loss.item() == 0.0
