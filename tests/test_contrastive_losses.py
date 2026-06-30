import torch

from tismir.losses import (
    audio_audio_supervised_contrastive,
    boundary_prediction_loss,
    cross_similarity_matching_loss,
    pairwise_probability_loss,
    pairwise_structure_relation_loss,
    pairwise_structure_loss,
    text_to_audio_infonce,
    token_uniformity_loss,
)


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


def test_pairwise_probability_loss_is_lower_for_correct_same_label_structure():
    targets = torch.tensor([[0, 0, 1, 1]])
    good_logits = torch.tensor([[[4.0, 0.0], [3.8, 0.2], [0.0, 4.0], [0.1, 3.9]]])
    bad_logits = torch.tensor([[[4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [0.0, 4.0]]])

    good_loss = pairwise_probability_loss(good_logits, targets)
    bad_loss = pairwise_probability_loss(bad_logits, targets)

    assert good_loss < bad_loss


def test_pairwise_probability_loss_handles_ignored_targets():
    logits = torch.tensor([[[4.0, 0.0], [3.8, 0.2], [0.0, 4.0]]])
    targets = torch.tensor([[0, -100, 1]])

    loss = pairwise_probability_loss(logits, targets)

    assert torch.isfinite(loss)


def test_pairwise_probability_loss_can_disable_class_balancing():
    logits = torch.tensor([[[4.0, 0.0], [3.8, 0.2], [0.0, 4.0], [0.1, 3.9]]])
    targets = torch.tensor([[0, 0, 1, 1]])

    balanced = pairwise_probability_loss(logits, targets, balance=True)
    unbalanced = pairwise_probability_loss(logits, targets, balance=False)

    assert torch.isfinite(balanced)
    assert torch.isfinite(unbalanced)


def test_pairwise_structure_loss_is_lower_for_correct_same_label_structure():
    targets = torch.tensor([[0, 0, 1, 1]])
    good_tokens = torch.tensor([[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]])
    bad_tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])

    good_loss = pairwise_structure_loss(good_tokens, targets)
    bad_loss = pairwise_structure_loss(bad_tokens, targets)

    assert good_loss < bad_loss


def test_pairwise_structure_loss_handles_ignored_targets():
    tokens = torch.tensor([[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]])
    targets = torch.tensor([[0, -100, 1]])

    loss = pairwise_structure_loss(tokens, targets)

    assert torch.isfinite(loss)


def test_pairwise_structure_relation_loss_is_lower_for_correct_relations():
    base_targets = torch.tensor([[0, 0, 0, 1]])
    segment_targets = torch.tensor([[0, 0, 1, 2]])
    good_logits = torch.tensor(
        [
            [
                [[0.0, 0.0, 4.0], [0.0, 0.0, 4.0], [0.0, 4.0, 0.0], [4.0, 0.0, 0.0]],
                [[0.0, 0.0, 4.0], [0.0, 0.0, 4.0], [0.0, 4.0, 0.0], [4.0, 0.0, 0.0]],
                [[0.0, 4.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0], [4.0, 0.0, 0.0]],
                [[4.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 4.0]],
            ]
        ]
    )
    bad_logits = good_logits.flip(dims=[-1])

    good_loss = pairwise_structure_relation_loss(good_logits, base_targets, segment_targets)
    bad_loss = pairwise_structure_relation_loss(bad_logits, base_targets, segment_targets)

    assert good_loss < bad_loss


def test_boundary_prediction_loss_is_lower_for_correct_boundaries():
    segment_targets = torch.tensor([[0, 0, 1, 1]])
    good_logits = torch.tensor([[-4.0, 4.0, -4.0]])
    bad_logits = -good_logits

    good_loss = boundary_prediction_loss(good_logits, segment_targets)
    bad_loss = boundary_prediction_loss(bad_logits, segment_targets)

    assert good_loss < bad_loss


def test_boundary_prediction_loss_handles_ignored_targets():
    segment_targets = torch.tensor([[0, -100, 1]])
    logits = torch.zeros((1, 2))

    loss = boundary_prediction_loss(logits, segment_targets)

    assert torch.isfinite(loss)
    assert loss == 0.0


def test_cross_similarity_matching_loss_is_lower_for_matching_similarity():
    targets = torch.tensor([[0, 1]])
    good_similarity = torch.tensor([[[0.9, 0.0], [0.1, 0.8]]])
    bad_similarity = torch.tensor([[[0.0, 0.9], [0.8, 0.1]]])

    good_loss = cross_similarity_matching_loss(good_similarity, targets)
    bad_loss = cross_similarity_matching_loss(bad_similarity, targets)

    assert good_loss < bad_loss


def test_cross_similarity_matching_loss_handles_ignored_targets():
    similarity = torch.tensor([[[0.9, 0.0], [0.1, 0.8]]])
    targets = torch.tensor([[0, -100]])

    loss = cross_similarity_matching_loss(similarity, targets)

    assert torch.isfinite(loss)


def test_token_uniformity_loss_is_lower_for_spread_tokens():
    collapsed = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    spread = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])

    collapsed_loss = token_uniformity_loss(collapsed)
    spread_loss = token_uniformity_loss(spread)

    assert spread_loss < collapsed_loss


def test_token_uniformity_loss_handles_mask():
    tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
    mask = torch.tensor([[True, False, True]])

    loss = token_uniformity_loss(tokens, mask=mask)

    assert torch.isfinite(loss)
