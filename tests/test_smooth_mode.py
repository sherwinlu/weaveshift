from weaveshift_two_tracks import TransitionCandidate, smooth_candidate_score
from weaveshift_plan import select_candidate


def cand(
    name,
    *,
    score=0.9,
    overlap=2.0,
    gain=1.0,
    cue=0.0,
    trim=1.0,
    collision=0.0,
    beat=1.0,
    energy=0.9,
):
    return TransitionCandidate(
        name=name,
        score=score,
        a_fade_start_sec=100.0,
        a_cut_sec=100.0 + overlap,
        b_cue_sec=cue,
        overlap_sec=overlap,
        b_gain_db=gain,
        trim_a_tail_sec=trim,
        vocal_collision_score=collision,
        beat_alignment_score=beat,
        energy_score=energy,
        placement_score=0.9,
        loudness_score=1.0 if abs(gain) <= 2.0 else 0.55,
        perceptual_loudness_score=0.9,
        compatibility_score=0.9,
        notes=["test"],
    )


def test_smooth_score_penalizes_forced_long_blend_with_large_gain():
    safe_short = cand("vocal_safe", score=0.98, overlap=2.0, gain=1.8, cue=0.0, trim=1.0)
    bad_long = cand("long_blend", score=0.73, overlap=8.0, gain=4.0, cue=6.0, trim=2.0)
    tasteful_smooth = cand("candidate", score=0.88, overlap=6.0, gain=1.2, cue=2.0, trim=2.0)

    assert smooth_candidate_score(tasteful_smooth) > smooth_candidate_score(bad_long)
    assert smooth_candidate_score(tasteful_smooth) > smooth_candidate_score(safe_short)


def test_smooth_penalizes_vocal_mud_in_long_overlap():
    clean_short = cand("quick_cut", score=0.86, overlap=2.0, gain=3.0, cue=4.0, trim=1.0, collision=0.0)
    muddy_long = cand("smooth", score=0.86, overlap=6.0, gain=4.0, cue=2.0, trim=1.0, collision=0.07)

    assert smooth_candidate_score(clean_short) > smooth_candidate_score(muddy_long)


def test_smooth_penalizes_loose_beat_alignment_on_long_overlap():
    beat_aligned = cand("smooth", score=0.80, overlap=8.0, gain=1.0, cue=1.0, trim=1.0, beat=1.0)
    loose_long = cand("smooth", score=0.80, overlap=8.0, gain=1.0, cue=1.0, trim=1.0, beat=0.0)

    assert smooth_candidate_score(beat_aligned) > smooth_candidate_score(loose_long)


def test_select_candidate_supports_smooth_mode():
    candidates = [
        cand("vocal_safe", score=0.98, overlap=2.0, gain=1.8, cue=0.0, trim=1.0),
        cand("smooth", score=0.86, overlap=6.0, gain=1.2, cue=2.0, trim=2.0),
    ]
    selected = select_candidate(candidates, "smooth", transition_index=1)
    assert selected.name == "smooth"
