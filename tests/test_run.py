from arborist.arborist import arborist


def test_run_small(simulate_small):
    """
    Test that the simulation runs without errors.
    """
    try:
        likelihoods, best_fit = arborist(
            simulate_small.candidate_set,
            simulate_small.read_counts,
            alpha=0.001,
            max_iter=10,
        )
        print(best_fit)

    except Exception as e:
        print(f"Arborist failed with error: {e}")
        assert False


# def test_run_large(simulate_large):
#     """
#     Test that the simulation runs without errors.
#     """
#     try:
#         likelihoods, best_fit= rank_trees(simulate_large.candidate_set, simulate_large.read_counts, alpha=0.001, max_iter=10)
#         print(best_fit)

#     except Exception as e:
#         print(f"Arborist failed with error: {e}")
#         assert False


def test_run_med(simulate_medium):
    """
    Test that the simulation runs without errors.
    """
    try:
        likelihoods, best_fit = arborist(
            simulate_medium.candidate_set,
            simulate_medium.read_counts,
            alpha=0.001,
            max_iter=10,
        )
        print(best_fit)

    except Exception as e:
        print(f"Arborist failed with error: {e}")
        assert False
