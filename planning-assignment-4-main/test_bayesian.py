# LLM Copilot did helped with some parts of this assignment see below 

import numpy as np
import pytest
from bayesian import StateGenerator, sample_observation, sample_transition, belief_update, belief_predict, initialize_belief

class TestBayesianInference:

    @pytest.mark.parametrize("initial_state,observation_list,prior_style", [
        (
            ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
            [(3,4)], "uniform",
        ),
        (
            ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
            [(3,4),(3,4)], "uniform",
        ),
        (
            ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
            [(3,4),(3,5)], "uniform",
        ),
        (
            ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
            [(3,5),(3,3),(3,4)], "uniform",
        ),
        (
            ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
            [(3,4)], "dirac",
        ),
        (
            ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
            [(3,4),(3,4)], "dirac",
        ),
        (
            ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
            [(3,4),(3,5)], "dirac",
        ),
        (
            ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
            [(3,5),(3,3),(3,4)], "dirac",
        ),
    ])

    # LLM Copilot suggested or assisted with parts of this code block
    def test_example_obserations(self, initial_state, observation_list,prior_style):
        """
        This test represents some sample test cases for you to test your Bayesian update;
        you should implement the tests here. Feel free to add additional parameters.
        """
        positions, (nrows, ncols) = initial_state
        true_c, true_r = positions[0]
        other_positions = set(positions[1:])

        belief = initialize_belief(initial_state, style=prior_style)
        assert belief.shape == (nrows, ncols)

        # Apply each observation
        for obs in observation_list:
            belief = belief_update(belief, obs, initial_state)
            assert belief.shape == (nrows, ncols)
            assert np.isclose(belief.sum(), 1.0)
            # Other pieces' cells must always be zero-probability.
            for (c, r) in other_positions:
                assert np.isclose(belief[r, c], 0.0)

        if prior_style == "dirac":
            # Dirac prior should stay a delta at the true cell since prior has
            # zero mass elsewhere.
            r_max, c_max = np.unravel_index(np.argmax(belief), belief.shape)
            assert (c_max, r_max) == (true_c, true_r)
            assert np.isclose(belief[true_r, true_c], 1.0)
        else:
            # Uniform prior should become non-uniform after at least one observation.
            flat = belief.ravel()
            assert not np.allclose(flat, np.full_like(flat, flat.mean()))
    # End of all suggested or assisted code in this block 

    @pytest.mark.parametrize("initial_state,action_list,prior_style", [
        (
            ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
            [(0,0)], "uniform",
        ),
        (
            ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
            [(0,0),(0,1)], "uniform",
        ),
        (
            ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
            [(1,0),(0,0)], "uniform",
        ),
        (
            ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
            [(0,0),(1,0),(-1,0)], "uniform",
        ),
        (
            ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
            [(0,0)], "dirac",
        ),
        (
            ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
            [(0,0),(0,1)], "dirac",
        ),
        (
            ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
            [(1,0),(0,0)], "dirac",
        ),
        (
            ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
            [(0,0),(1,0),(-1,0)], "dirac",
        ),
    ])

    # LLM Copilot suggested or assisted with parts of this code block
    def test_example_actions(self, initial_state, action_list,prior_style):
        positions, (nrows, ncols) = initial_state
        true_start = positions[0]
        other_positions = set(positions[1:])

        belief = initialize_belief(initial_state, style=prior_style)

        for action in action_list:
            old_belief = belief.copy()
            belief = belief_predict(belief, action, initial_state)
            assert belief.shape == (nrows, ncols)
            assert np.isclose(belief.sum(), 1.0)
            for (c, r) in other_positions:
                assert np.isclose(belief[r, c], 0.0)

            # (0,0) is a no-op: belief should not change.
            if action == (0, 0):
                assert np.allclose(belief, old_belief)

        if prior_style == "dirac":
            # Compute the exact final position using the transition model and
            # make sure the belief is a delta at that cell.
            expected_pos = true_start
            for action in action_list:
                s = ([expected_pos] + positions[1:], (nrows, ncols))
                new_pos, _ = sample_transition(s, action)
                assert new_pos is not None  # all parameterized actions are valid
                expected_pos = new_pos

            r_max, c_max = np.unravel_index(np.argmax(belief), belief.shape)
            assert (c_max, r_max) == expected_pos
            assert np.isclose(belief[expected_pos[1], expected_pos[0]], 1.0)  
    # End of all suggested or assisted code in this block    
