import numpy as np
import pytest
from game import BoardState, GameSimulator, AdversarialSearchPlayer, PlayerWithAlgorithmB
from search import GameStateProblem

class TestGame:

    @pytest.mark.parametrize(
        "p1_class, p2_class, encoded_state_tuple, exp_winner, exp_stat",
        [
            (AdversarialSearchPlayer, PlayerWithAlgorithmB, (49, 37, 46,  7, 55,  7, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
            (AdversarialSearchPlayer, PlayerWithAlgorithmB, (49, 37, 46,  0, 55,  0, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
            (PlayerWithAlgorithmB, AdversarialSearchPlayer, (14, 21, 22, 28, 29, 22,  9, 20, 34, 39, 55, 55), "BLACK", "No issues"),
            (PlayerWithAlgorithmB, AdversarialSearchPlayer, (14, 21, 22, 28, 29, 22, 11, 20, 34, 39, 55, 55), "BLACK", "No issues"),
            (AdversarialSearchPlayer, PlayerWithAlgorithmB, (44, 37, 46, 34, 40, 34,  1,  2, 52,  4,  5, 52), "WHITE", "No issues"),
            (AdversarialSearchPlayer, PlayerWithAlgorithmB, (44, 37, 46, 28, 40, 28,  1,  2, 52,  4,  5, 52), "WHITE", "No issues"),
        ]
    )
    def test_adversarial_search(self, p1_class, p2_class, encoded_state_tuple, exp_winner, exp_stat):
        b1 = BoardState()
        b1.state = np.array(encoded_state_tuple)
        b1.decode_state = b1.make_state()
        gsp1 = GameStateProblem(b1, b1, 0)
        gsp2 = GameStateProblem(b1, b1, 1)
        players = [p1_class(gsp1, 0), p2_class(gsp2, 1)]
        sim = GameSimulator(players)
        sim.game_state = b1
        rounds, winner, status = sim.run()
        assert winner == exp_winner and status == exp_stat
