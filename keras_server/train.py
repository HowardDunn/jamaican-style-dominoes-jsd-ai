"""Pure Python reinforcement training loop for domino AI — port of jsd-ai/main.go trainReinforced()."""

from __future__ import annotations

import os
import random
import time

import numpy as np

from domino_game import (
    INDEX_SUIT_MAP, DOUBLE_SIX, EMPTY_DOMINO,
    LEFT, RIGHT, POSED,
    POSED_CARD, PLAYED_CARD, PASSED, PLAYER_TURN, NULL_EVENT,
    ROUND_WIN, GAME_WIN, ROUND_DRAW, ROUND_BLOCKED, ROUND_INVALID,
    get_suits, is_double, get_count,
    Board, GameEvent, RandomPlayer, LocalGame, rotate_game_event,
)
from model import DominoModel

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")


# ---------------------------------------------------------------------------
# KerasNNPlayer — port of nn.KerasPlayer / nn.JSDNN play + feature logic
# ---------------------------------------------------------------------------

class KerasNNPlayer:
    """Domino player backed by a Keras DominoModel for in-process GPU inference."""

    def __init__(self, model_id: str, hidden_dims: list[int], epsilon: float = 0.1):
        self.model_id = model_id
        self.hidden_dims = hidden_dims
        self.model = DominoModel(input_dim=126, hidden_dims=hidden_dims, output_dim=56)
        self.hand: list[int] = []
        self.pass_memory = np.zeros(28, dtype=np.float64)
        self.known_not_haves: list[dict[int, bool]] = [{i: False for i in range(28)} for _ in range(4)]
        self.epsilon = epsilon
        self.total_wins = 0
        self.total_wins2 = 0
        self.random_mode = False  # for ComputerPlayer fallback

    # -- Player protocol --

    def get_hand(self) -> list[int]:
        return self.hand

    def set_hand(self, hand: list[int]) -> None:
        self.hand = list(hand)

    def pose_double_six(self) -> int:
        return DOUBLE_SIX

    def pose_card(self) -> int:
        """Suit-counting strategy (port of JSDNN.PoseCard)."""
        suit_count = [0] * 7
        for c in self.hand:
            if c == EMPTY_DOMINO or c >= 28:
                continue
            s1, _ = get_suits(c)
            if is_double(c):
                suit_count[s1] += 1

        for c in self.hand:
            if c == EMPTY_DOMINO or c >= 28:
                continue
            s1, s2 = get_suits(c)
            suit_sum = suit_count[s1] + suit_count[s2]
            if not is_double(c) and suit_sum > 0:
                suit_count[s1] += 1
                suit_count[s2] += 1

        max_suit = 0
        max_count = 0
        for i in range(7):
            if suit_count[i] >= max_count:
                max_count = suit_count[i]
                max_suit = i

        for c in self.hand:
            if c == EMPTY_DOMINO or c >= 28:
                continue
            s1, _ = get_suits(c)
            if is_double(c) and s1 == max_suit:
                return c

        # Fallback: first double, then first card
        for c in self.hand:
            if c == EMPTY_DOMINO or c >= 28:
                continue
            if is_double(c):
                return c
        return self.hand[0]

    def count_suit(self, suit: int) -> int:
        count = 0
        for c in self.hand:
            if c == EMPTY_DOMINO or c >= 28:
                continue
            s1, s2 = get_suits(c)
            if s1 == suit or s2 == suit:
                count += 1
        return count

    def play_card(self, game_event: GameEvent) -> tuple[int, str]:
        rotated = rotate_game_event(game_event, game_event.player)

        # Epsilon-greedy exploration
        if self.epsilon > 0 and random.random() < self.epsilon:
            compatible = []
            for card in rotated.player_hands[rotated.player]:
                if card >= 28 or card == EMPTY_DOMINO:
                    continue
                s1, s2 = get_suits(card)
                if rotated.board_state and rotated.board_state.card_posed:
                    if s1 == rotated.board_state.left_suit or s2 == rotated.board_state.left_suit:
                        compatible.append((card, LEFT))
                    if s1 == rotated.board_state.right_suit or s2 == rotated.board_state.right_suit:
                        compatible.append((card, RIGHT))
            if compatible:
                return random.choice(compatible)

        # Model prediction
        features = self.convert_game_event_to_features(rotated)
        valid_mask = self.get_output_mask(rotated)
        features_np = np.array(features, dtype=np.float32)
        mask_np = np.array(valid_mask, dtype=np.float32)

        card, side, _ = self.model.predict(features_np, mask_np)

        # Validate
        if card > 27:
            return self._fallback_play(rotated)
        s1, s2 = get_suits(card)
        ok = False
        if side == LEFT:
            if s1 == rotated.board_state.left_suit or s2 == rotated.board_state.left_suit:
                ok = True
        else:
            if s1 == rotated.board_state.right_suit or s2 == rotated.board_state.right_suit:
                ok = True
        if not ok:
            return self._fallback_play(rotated)
        return card, side

    def _fallback_play(self, game_event: GameEvent) -> tuple[int, str]:
        """Greedy fallback like ComputerPlayer."""
        board = game_event.board_state
        hand = game_event.player_hands[game_event.player] if game_event.player_hands else self.hand
        # Doubles first
        for c in hand:
            if c == EMPTY_DOMINO or c >= 28:
                continue
            if not is_double(c):
                continue
            s1, s2 = get_suits(c)
            if s1 == board.left_suit or s2 == board.left_suit:
                return c, LEFT
            if s1 == board.right_suit or s2 == board.right_suit:
                return c, RIGHT
        for c in hand:
            if c == EMPTY_DOMINO or c >= 28:
                continue
            s1, s2 = get_suits(c)
            if s1 == board.left_suit or s2 == board.left_suit:
                return c, LEFT
            if s1 == board.right_suit or s2 == board.right_suit:
                return c, RIGHT
        return EMPTY_DOMINO, LEFT

    # -- Pass memory --

    def update_pass_memory(self, game_event: GameEvent) -> None:
        rotated = rotate_game_event(game_event, game_event.player)
        card = rotated.card
        player = rotated.player
        s1, s2 = get_suits(card)
        s1_idx = player * 7 + s1
        s2_idx = player * 7 + s2
        self.pass_memory[s1_idx] = 1.0
        self.pass_memory[s2_idx] = 1.0
        for i in range(28):
            cs1, cs2 = get_suits(i)
            if cs1 == s1 or cs1 == s2:
                self.known_not_haves[player][i] = True
            elif cs2 == s1 or cs2 == s2:
                self.known_not_haves[player][i] = True

    def reset_pass_memory(self) -> None:
        self.pass_memory = np.zeros(28, dtype=np.float64)
        self.known_not_haves = [{i: False for i in range(28)} for _ in range(4)]

    # -- Feature encoding (126-float vector) --

    def convert_game_event_to_features(self, game_event: GameEvent) -> list[float]:
        player_hand = [0.0] * 28
        board_state = [0.0] * 28
        suit_state = [0.0] * 14
        card_remaining = [0.0] * 28

        bs = game_event.board_state
        ph = game_event.player_hands[game_event.player] if game_event.player_hands else []

        for card in ph:
            if card < 28:
                s1, s2 = get_suits(card)
                card_compatible = False
                if s1 == bs.left_suit or s2 == bs.left_suit:
                    card_compatible = True
                if s1 == bs.right_suit or s2 == bs.right_suit:
                    card_compatible = True
                if card_compatible:
                    player_hand[card] = 1.0

        if game_event.event_type in (PLAYED_CARD, POSED_CARD):
            if bs.card_posed:
                player_hand[game_event.card] = 1.0

        if bs.card_posed:
            board_state[bs.posed_card] = 1.0
            for c in bs.left_board:
                board_state[c] = 1.0
            for c in bs.right_board:
                board_state[c] = 1.0
            suit_state[bs.left_suit] = 1.0
            suit_state[bs.right_suit + 7] = 1.0

        cr0 = game_event.player_cards_remaining[0] - 1
        cr1 = game_event.player_cards_remaining[1] + 6
        cr2 = game_event.player_cards_remaining[2] + 13
        cr3 = game_event.player_cards_remaining[3] + 20
        for cr in (cr0, cr1, cr2, cr3):
            if cr < 0:
                cr = 0
            if cr < 28:
                card_remaining[cr] = 1.0

        features = player_hand + board_state + suit_state + list(self.pass_memory) + card_remaining
        return features

    def get_output_mask(self, game_event: GameEvent) -> list[float]:
        """Side-specific valid action mask (56 outputs: 0-27 left, 28-55 right)."""
        mask = [0.0] * 56
        bs = game_event.board_state
        ph = game_event.player_hands[game_event.player] if game_event.player_hands else []

        for card in ph:
            if card < 28:
                s1, s2 = get_suits(card)
                if s1 == bs.left_suit or s2 == bs.left_suit:
                    mask[card] = 1.0
                if s1 == bs.right_suit or s2 == bs.right_suit:
                    mask[card + 28] = 1.0

        if game_event.event_type in (PLAYED_CARD, POSED_CARD):
            if game_event.card < 28:
                s1, s2 = get_suits(game_event.card)
                if s1 == bs.left_suit or s2 == bs.left_suit:
                    mask[game_event.card] = 1.0
                if s1 == bs.right_suit or s2 == bs.right_suit:
                    mask[game_event.card + 28] = 1.0
        return mask

    # -- Reward computation (port of ConvertCardChoiceToTensorReinforced) --

    def compute_reinforced_target(self, game_event: GameEvent, next_events: list[GameEvent | None]) -> list[float]:
        target = [0.0] * 56
        index = game_event.card
        suit_played = game_event.board_state.left_suit
        if game_event.side == RIGHT:
            index += 28
            suit_played = game_event.board_state.right_suit

        reward = 0.0
        chain_broken = False
        board_suit_count = game_event.board_state.count_suit(suit_played)
        hand_suit_count = self.count_suit(suit_played)
        has_hard_end = (board_suit_count + hand_suit_count == 6)
        is_dbl = is_double(game_event.card)
        won = False
        won_by_block = True

        for i, ne in enumerate(next_events):
            if ne is None or i > 4:
                break
            if ne.event_type == PASSED:
                if ne.player == 0:
                    reward = -1.0
                elif not chain_broken:
                    reward += 1.0
            elif ne.event_type == PLAYED_CARD:
                if ne.player != 0:
                    chain_broken = True
            elif ne.event_type == ROUND_WIN:
                if ne.player == 0:
                    reward = 7.0
                    won = True
                else:
                    reward = -7.0
                    won_by_block = False
                for j in range(len(ne.player_cards_remaining)):
                    if ne.player_cards_remaining[j] == 0:
                        won_by_block = False
                        break
            elif ne.event_type == ROUND_DRAW:
                won_by_block = False

        if not won:
            won_by_block = False
        if has_hard_end and not won:
            reward = -5.0
        elif is_dbl and not won:
            reward += 3.0
        if won_by_block:
            reward *= 1.5

        target[index] = reward
        return target

    # -- Weight persistence --

    def save_weights(self, path: str = "") -> None:
        if not path:
            os.makedirs(WEIGHTS_DIR, exist_ok=True)
            path = os.path.join(WEIGHTS_DIR, f"{self.model_id}.weights.h5")
        self.model.save_weights(path)

    def load_weights(self, path: str = "") -> None:
        if not path:
            path = os.path.join(WEIGHTS_DIR, f"{self.model_id}.weights.h5")
        if os.path.exists(path):
            self.model.load_weights(path)


# ---------------------------------------------------------------------------
# Training loop — port of trainReinforced() / trainKerasReinforced()
# ---------------------------------------------------------------------------

def train_reinforced():
    start = time.time()
    same_game_iterations = 8
    max_games = 100
    total_round_wins = [0, 0, 0, 0]

    # Create 4 NN players with different architectures
    kp1 = KerasNNPlayer("keras1", [150])
    kp2 = KerasNNPlayer("keras2", [64, 64])
    kp3 = KerasNNPlayer("keras3", [32, 32])
    kp4 = KerasNNPlayer("keras4", [128, 128])

    # Try loading existing weights
    kp1.load_weights()
    kp2.load_weights()
    kp3.load_weights()
    kp4.load_weights()

    kp1.epsilon = 0.1
    kp2.epsilon = 0.1
    kp3.epsilon = 0.1
    kp4.epsilon = 0.1

    def run_iter(players_list, kp_players, tot_wins, rand_seed):
        """Run a single game to completion, collecting and training on round events."""
        game = LocalGame(players_list, random_seed=rand_seed, game_type="cutthroat")
        round_game_events: list[GameEvent] = []
        last_event = GameEvent()

        while last_event is not None and last_event.event_type != GAME_WIN:
            last_event = game.advance()
            rotated_0 = rotate_game_event(last_event, 0)

            if rotated_0.event_type in (PLAYED_CARD, PASSED) or \
               last_event.event_type in (ROUND_WIN, ROUND_DRAW):
                round_game_events.append(rotated_0)

            if last_event.event_type in (ROUND_WIN, ROUND_DRAW):
                # Reset pass memory
                kp1.reset_pass_memory()
                kp2.reset_pass_memory()
                kp3.reset_pass_memory()
                kp4.reset_pass_memory()

                # Collect batches per player
                player_batches: list[list[tuple]] = [[], [], [], []]
                player_learn_rates = [0.0] * 4
                for p in range(4):
                    player_learn_rates[p] = 0.0001
                    if len(kp_players[p].hidden_dims) == 1:
                        player_learn_rates[p] = 0.001

                for i, ge in enumerate(round_game_events):
                    if ge.event_type == PLAYED_CARD:
                        kp = kp_players[ge.player]
                        rotated_ge = rotate_game_event(ge, ge.player)
                        next_relevant: list[GameEvent | None] = [None] * 16
                        for j in range(1, 17):
                            if (i + j) >= len(round_game_events):
                                break
                            next_relevant[j - 1] = rotate_game_event(
                                round_game_events[i + j], ge.player
                            )

                        features = kp.convert_game_event_to_features(rotated_ge)
                        # Check if only one valid choice -> skip
                        mask_vals = kp.get_output_mask(rotated_ge)
                        compat_l = sum(1 for v in mask_vals[:28] if v > 0)
                        compat_r = sum(1 for v in mask_vals[28:] if v > 0)
                        if (compat_l + compat_r) <= 1:
                            continue

                        target = kp.compute_reinforced_target(rotated_ge, next_relevant)
                        action_mask = [0.0] * 56
                        action_idx = rotated_ge.card
                        if rotated_ge.side == RIGHT:
                            action_idx += 28
                        action_mask[action_idx] = 1.0
                        player_batches[ge.player].append((features, target, action_mask))

                    elif ge.event_type == PASSED:
                        kp_players[ge.player].update_pass_memory(ge)

                    elif ge.event_type in (ROUND_WIN, ROUND_DRAW):
                        kp_players[ge.player].reset_pass_memory()

                # Train each player's model with collected batch
                for p in range(4):
                    batch = player_batches[p]
                    if not batch:
                        continue
                    fb = np.array([s[0] for s in batch], dtype=np.float32)
                    tb = np.array([s[1] for s in batch], dtype=np.float32)
                    mb = np.array([s[2] for s in batch], dtype=np.float32)
                    kp_players[p].model.train_batch(fb, tb, mb, player_learn_rates[p])

                round_game_events = []

        # Accumulate wins
        for i in range(4):
            tot_wins[i] += last_event.player_wins[i]

        # Save weights after each game
        kp1.save_weights()
        kp2.save_weights()
        kp3.save_weights()
        kp4.save_weights()

    def reinforcement_learn():
        # Phase 1: Same-seat games
        for _ in range(same_game_iterations):
            kp_players = [kp1, kp2, kp3, kp4]
            players_list = list(kp_players)
            tot_wins = [0, 0, 0, 0]
            # Use per-player total_wins references
            seed = random.randint(0, 2**62)
            run_iter(players_list, kp_players, tot_wins, seed)
            kp1.total_wins += tot_wins[0]
            kp2.total_wins += tot_wins[1]
            kp3.total_wins += tot_wins[2]
            kp4.total_wins += tot_wins[3]

        # Phase 2: Shuffled-seat games
        for _ in range(same_game_iterations):
            kp_players = [kp1, kp2, kp3, kp4]
            random.shuffle(kp_players)
            players_list = list(kp_players)
            tot_wins = [0, 0, 0, 0]
            seed = random.randint(0, 2**62)
            run_iter(players_list, kp_players, tot_wins, seed)
            for i, kp in enumerate(kp_players):
                kp.total_wins2 += tot_wins[i]

        # Phase 3: Benchmark vs random (no exploration)
        saved_eps = [kp1.epsilon, kp2.epsilon, kp3.epsilon, kp4.epsilon]
        kp1.epsilon = 0
        kp2.epsilon = 0
        kp3.epsilon = 0
        kp4.epsilon = 0

        kp_players = [kp1, kp2, kp3, kp4]
        bench_players: list = [
            RandomPlayer(random_mode=True),
            RandomPlayer(random_mode=True),
            kp_players[2],
            RandomPlayer(random_mode=True),
        ]
        # Shuffle positions
        indices = list(range(4))
        random.shuffle(indices)
        shuffled_bench = [None] * 4
        shuffled_kps = [None] * 4
        shuffled_tots = [0, 0, 0, 0]
        for new_i, old_i in enumerate(indices):
            shuffled_bench[new_i] = bench_players[old_i]
            shuffled_kps[new_i] = kp_players[old_i]

        for _ in range(same_game_iterations):
            seed = random.randint(0, 2**62)
            run_iter(shuffled_bench, shuffled_kps, shuffled_tots, seed)

        for new_i, old_i in enumerate(indices):
            total_round_wins[old_i] += shuffled_tots[new_i]

        kp1.epsilon = saved_eps[0]
        kp2.epsilon = saved_eps[1]
        kp3.epsilon = saved_eps[2]
        kp4.epsilon = saved_eps[3]

    # Main training loop
    print(f"Starting training with seed {int(time.time())}")
    for j in range(max_games):
        reinforcement_learn()
        total = sum(total_round_wins)
        if total == 0:
            total = 1
        print(f"\n=== Iteration {j + 1}/{max_games} ===")
        print(f"Benchmark vs Random wins: {total_round_wins}  Total rounds: {total}")
        print(f"  Ratios: [{total_round_wins[0]/total:.3f}, {total_round_wins[1]/total:.3f}, "
              f"{total_round_wins[2]/total:.3f}, {total_round_wins[3]/total:.3f}]")

        tw = kp1.total_wins + kp2.total_wins + kp3.total_wins + kp4.total_wins
        if tw == 0:
            tw = 1
        print(f"NN Wins (same-seat): [{kp1.total_wins}, {kp2.total_wins}, {kp3.total_wins}, {kp4.total_wins}]")
        print(f"  Ratios: [{kp1.total_wins/tw:.3f}, {kp2.total_wins/tw:.3f}, "
              f"{kp3.total_wins/tw:.3f}, {kp4.total_wins/tw:.3f}]")

        tw2 = kp1.total_wins2 + kp2.total_wins2 + kp3.total_wins2 + kp4.total_wins2
        if tw2 == 0:
            tw2 = 1
        print(f"NN Wins (shuffled): [{kp1.total_wins2}, {kp2.total_wins2}, {kp3.total_wins2}, {kp4.total_wins2}]")
        print(f"  Ratios: [{kp1.total_wins2/tw2:.3f}, {kp2.total_wins2/tw2:.3f}, "
              f"{kp3.total_wins2/tw2:.3f}, {kp4.total_wins2/tw2:.3f}]")

    kp1.save_weights()
    kp2.save_weights()
    kp3.save_weights()
    kp4.save_weights()
    elapsed = time.time() - start
    print(f"\nTraining complete. Took {elapsed:.1f}s")


if __name__ == "__main__":
    train_reinforced()
