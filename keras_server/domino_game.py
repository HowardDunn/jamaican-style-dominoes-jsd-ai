"""Pure Python domino game engine — port of go-dominos/dominos/*.go + rotation utils."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

# ---------------------------------------------------------------------------
# Domino constants
# ---------------------------------------------------------------------------

INDEX_SUIT_MAP: dict[int, tuple[int, int]] = {
    0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (2, 0), 4: (2, 1), 5: (2, 2),
    6: (3, 0), 7: (3, 1), 8: (3, 2), 9: (3, 3), 10: (4, 0), 11: (4, 1),
    12: (4, 2), 13: (4, 3), 14: (4, 4), 15: (5, 0), 16: (5, 1), 17: (5, 2),
    18: (5, 3), 19: (5, 4), 20: (5, 5), 21: (6, 0), 22: (6, 1), 23: (6, 2),
    24: (6, 3), 25: (6, 4), 26: (6, 5), 27: (6, 6),
}

DOUBLE_SIX = 27
EMPTY_DOMINO = 29

LEFT = "left"
RIGHT = "right"
POSED = "posed"


def get_suits(card: int) -> tuple[int, int]:
    return INDEX_SUIT_MAP[card]


def is_double(card: int) -> bool:
    s1, s2 = INDEX_SUIT_MAP[card]
    return s1 == s2


def get_count(card: int) -> int:
    s1, s2 = INDEX_SUIT_MAP[card]
    return s1 + s2


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

POSED_CARD = "posedCard"
PLAYED_CARD = "playedCard"
PASSED = "passed"
PLAYER_TURN = "playerTurn"
NULL_EVENT = "null"
ROUND_WIN = "roundWin"
GAME_WIN = "gameWin"
ROUND_DRAW = "roundDraw"
ROUND_BLOCKED = "roundBlocked"
ROUND_INVALID = "roundInvalid"
SHUFFLE = "shuffle"

# Game states
DSP = "dsp"
EPT = "ept"
EPP = "epp"
EPPDS = "eppds"
CRB = "crb"
CRW = "crw"
WP = "wp"
GWC = "gwc"
IDLE = "idle"


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------

@dataclass
class Board:
    card_posed: bool = False
    posed_card: int = 0
    left_board: list[int] = field(default_factory=list)
    right_board: list[int] = field(default_factory=list)
    left_suit: int = 0
    right_suit: int = 0

    def copy(self) -> Board:
        return Board(
            card_posed=self.card_posed,
            posed_card=self.posed_card,
            left_board=list(self.left_board),
            right_board=list(self.right_board),
            left_suit=self.left_suit,
            right_suit=self.right_suit,
        )

    def count_suit(self, suit: int) -> int:
        count = 0
        all_cards = list(self.left_board) + list(self.right_board)
        if self.card_posed:
            all_cards.append(self.posed_card)
        for card_idx in all_cards:
            s1, s2 = get_suits(card_idx)
            if s1 == suit or s2 == suit:
                count += 1
        return count


# ---------------------------------------------------------------------------
# GameEvent
# ---------------------------------------------------------------------------

@dataclass
class GameEvent:
    event_type: str = NULL_EVENT
    player: int = 0
    player_cards_remaining: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    player_wins: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    player_names: list[str] = field(default_factory=lambda: ["You", "Sammy", "Waz", "Riles"])
    player_hands: list[list[int]] = field(default_factory=lambda: [[], [], [], []])
    card: int = 0
    side: str = LEFT
    board_state: Board | None = None
    game_state: str = ""
    pose_mode_double_six: bool = False
    won_key: bool = False


# ---------------------------------------------------------------------------
# Player protocol
# ---------------------------------------------------------------------------

class Player(Protocol):
    def get_hand(self) -> list[int]: ...
    def set_hand(self, hand: list[int]) -> None: ...
    def play_card(self, game_event: GameEvent) -> tuple[int, str]: ...
    def pose_double_six(self) -> int: ...
    def pose_card(self) -> int: ...


# ---------------------------------------------------------------------------
# RandomPlayer (port of ComputerPlayer)
# ---------------------------------------------------------------------------

class RandomPlayer:
    def __init__(self, random_mode: bool = False):
        self.hand: list[int] = []
        self.random_mode = random_mode

    def get_hand(self) -> list[int]:
        return self.hand

    def set_hand(self, hand: list[int]) -> None:
        self.hand = list(hand)

    def count_suit(self, suit: int) -> int:
        count = 0
        for card_idx in self.hand:
            if card_idx == EMPTY_DOMINO or card_idx >= 28:
                continue
            s1, s2 = get_suits(card_idx)
            if s1 == suit or s2 == suit:
                count += 1
        return count

    def play_card(self, game_event: GameEvent) -> tuple[int, str]:
        board = game_event.board_state
        hand = list(self.hand)
        if self.random_mode:
            random.shuffle(hand)

        # First play doubles
        for card_idx in hand:
            if card_idx == EMPTY_DOMINO or card_idx >= 28:
                continue
            if not is_double(card_idx):
                continue
            s1, s2 = get_suits(card_idx)
            if s1 == board.left_suit or s2 == board.left_suit:
                return card_idx, LEFT
            if s1 == board.right_suit or s2 == board.right_suit:
                return card_idx, RIGHT

        # Then any matching card
        for card_idx in hand:
            if card_idx == EMPTY_DOMINO or card_idx >= 28:
                continue
            s1, s2 = get_suits(card_idx)
            if s1 == board.left_suit or s2 == board.left_suit:
                return card_idx, LEFT
            if s1 == board.right_suit or s2 == board.right_suit:
                return card_idx, RIGHT

        return EMPTY_DOMINO, LEFT

    def pose_double_six(self) -> int:
        return DOUBLE_SIX

    def pose_card(self) -> int:
        for card_idx in self.hand:
            if card_idx == EMPTY_DOMINO or card_idx >= 28:
                continue
            if is_double(card_idx):
                return card_idx
        return self.hand[0]


# ---------------------------------------------------------------------------
# Rotation utils (port of jsd-online-game/game/utils.go)
# ---------------------------------------------------------------------------

def _rotate_list4(lst: list, amount: int) -> list:
    """Rotate a 4-element list left by `amount`."""
    amount = amount % 4
    return lst[amount:] + lst[:amount]


def rotate_game_event(event: GameEvent, amount: int) -> GameEvent:
    """Port of CopyandRotateGameEvent."""
    players = [0, 1, 2, 3]
    rotated = _rotate_list4(players, amount)

    def find_index(num: int) -> int:
        for i in range(4):
            if rotated[i] == num:
                return i
        return 0

    return GameEvent(
        event_type=event.event_type,
        player=find_index(event.player),
        player_cards_remaining=_rotate_list4(list(event.player_cards_remaining), amount),
        player_wins=_rotate_list4(list(event.player_wins), amount),
        player_names=_rotate_list4(list(event.player_names), amount),
        player_hands=_rotate_list4([list(h) for h in event.player_hands], amount),
        card=event.card,
        side=event.side,
        board_state=event.board_state,
        game_state=event.game_state,
        pose_mode_double_six=event.pose_mode_double_six,
        won_key=event.won_key,
    )


# ---------------------------------------------------------------------------
# LocalGame — full state machine (port of game.go)
# ---------------------------------------------------------------------------

class LocalGame:
    def __init__(self, players: list, random_seed: int = 0, game_type: str = "cutthroat"):
        if game_type not in ("cutthroat", "partner"):
            game_type = "cutthroat"
        self.game_type = game_type
        self.board = Board()
        self.state = DSP
        self.players: list = players  # 4 Player-like objects
        self.player_wins = [0, 0, 0, 0]
        self.random_seed = random_seed
        self.player_turn = 0
        self.last_winner = 0
        self.num_rounds = 0
        self.pose_mode_double_six = False
        self.last_domino_played = 0
        self.pass_count = 0

    def reset(self, random_seed: int, game_type: str) -> None:
        self.random_seed = random_seed
        self.state = DSP
        self.player_wins = [0, 0, 0, 0]
        self.num_rounds = 0
        self.game_type = game_type
        self.board = Board()

    # -- helpers --

    def shuffle_and_distribute(self) -> None:
        all_dominos = list(range(28))
        rng = random.Random(self.random_seed + self.num_rounds)
        rng.shuffle(all_dominos)
        self.players[0].set_hand(all_dominos[0:7])
        self.players[1].set_hand(all_dominos[7:14])
        self.players[2].set_hand(all_dominos[14:21])
        self.players[3].set_hand(all_dominos[21:28])

    def _find_double_six_player(self) -> int:
        for i in range(4):
            hand = self.players[i].get_hand()
            for card in hand:
                if card == DOUBLE_SIX:
                    return i
        return -1

    def _advance_player_turn(self) -> None:
        self.player_turn = (self.player_turn + 1) % 4

    def update_board(self, new_card: int, side: str) -> None:
        s1, s2 = get_suits(new_card)
        if side == POSED:
            self.board.card_posed = True
            self.board.posed_card = new_card
            self.board.left_suit = s1
            self.board.right_suit = s2
        elif side == LEFT:
            if self.board.left_suit == s1:
                self.board.left_suit = s2
            elif self.board.left_suit == s2:
                self.board.left_suit = s1
            self.board.left_board.append(new_card)
        elif side == RIGHT:
            if self.board.right_suit == s1:
                self.board.right_suit = s2
            elif self.board.right_suit == s2:
                self.board.right_suit = s1
            self.board.right_board.append(new_card)

    def _get_player_num_cards_remaining(self, player: int) -> int:
        count = 0
        for card in self.players[player].get_hand():
            if card != EMPTY_DOMINO and card < 28:
                count += 1
        return count

    def _get_players_num_cards_remaining(self) -> list[int]:
        return [self._get_player_num_cards_remaining(i) for i in range(4)]

    def _get_player_hand_count(self, player: int) -> int:
        """Sum of pip counts for remaining cards in hand."""
        total = 0
        for card in self.players[player].get_hand():
            if card != EMPTY_DOMINO and card < 28:
                total += get_count(card)
        return total

    def _remove_card_from_player(self, card: int, player: int) -> None:
        hand = self.players[player].get_hand()
        for i in range(len(hand)):
            if hand[i] == card:
                hand[i] = EMPTY_DOMINO
        self.players[player].set_hand(hand)

    def _player_has_card(self, player: int, card: int) -> bool:
        for c in self.players[player].get_hand():
            if c == EMPTY_DOMINO:
                continue
            if c == card:
                return True
        return False

    def _can_play_card(self, player: int) -> bool:
        for card in self.players[player].get_hand():
            if card == EMPTY_DOMINO:
                continue
            s1, s2 = get_suits(card)
            if s1 == self.board.left_suit or s1 == self.board.right_suit:
                return True
            if s2 == self.board.left_suit or s2 == self.board.right_suit:
                return True
        return False

    def _won_key(self, card: int) -> bool:
        if is_double(card):
            return False
        pk1, pk2 = get_suits(card)

        def get_counts(eval_suit: int) -> int:
            count = 0
            ps1, ps2 = get_suits(self.board.posed_card)
            if ps1 == eval_suit or ps2 == eval_suit:
                count += 1
            for ci in self.board.left_board:
                cs1, cs2 = get_suits(ci)
                if cs1 == eval_suit or cs2 == eval_suit:
                    count += 1
            for ci in self.board.right_board:
                cs1, cs2 = get_suits(ci)
                if cs1 == eval_suit or cs2 == eval_suit:
                    count += 1
            return count

        if get_counts(pk1) != 7 or get_counts(pk2) != 7:
            return False
        if self.board.left_suit == pk1 and self.board.right_suit == pk1:
            return True
        if self.board.left_suit == pk2 and self.board.right_suit == pk2:
            return True
        return False

    def _is_game_blocked(self) -> bool:
        potential_suit = self.board.left_suit
        if self.board.left_suit != self.board.right_suit:
            return False
        suit_on_board_count = 0
        ps1, ps2 = get_suits(self.board.posed_card)
        if ps1 == potential_suit or ps2 == potential_suit:
            suit_on_board_count += 1
        for ci in self.board.left_board:
            cs1, cs2 = get_suits(ci)
            if cs1 == potential_suit or cs2 == potential_suit:
                suit_on_board_count += 1
        for ci in self.board.right_board:
            cs1, cs2 = get_suits(ci)
            if cs1 == potential_suit or cs2 == potential_suit:
                suit_on_board_count += 1
        return suit_on_board_count == 7

    def _player_names(self) -> list[str]:
        return ["You", "Sammy", "Waz", "Riles"]

    # -- main state machine --

    def advance(self) -> GameEvent:
        if self.state == WP:
            # WinnerPose
            self.board.card_posed = False
            self.board.left_board = []
            self.board.right_board = []
            self.pose_mode_double_six = False
            self.shuffle_and_distribute()
            self.player_turn = self.last_winner
            evt = GameEvent(
                event_type=PLAYER_TURN, player=self.player_turn,
                board_state=Board(), player_wins=list(self.player_wins),
                pose_mode_double_six=self.pose_mode_double_six,
                player_cards_remaining=self._get_players_num_cards_remaining(),
                game_state=self.state,
            )
            self.state = EPP
            return evt

        if self.state == EPP:
            # ExecutePlayerPose
            self.player_turn = self.last_winner
            card = self.players[self.player_turn].pose_card()
            if card == EMPTY_DOMINO or not self._player_has_card(self.player_turn, card):
                return GameEvent(
                    event_type=NULL_EVENT, player_wins=list(self.player_wins),
                    player_cards_remaining=self._get_players_num_cards_remaining(),
                )
            self.update_board(card, POSED)
            self._remove_card_from_player(card, self.player_turn)
            evt = GameEvent(
                event_type=POSED_CARD, player=self.player_turn,
                card=self.board.posed_card, side=POSED,
                board_state=Board(), player_wins=list(self.player_wins),
                pose_mode_double_six=self.pose_mode_double_six,
                player_cards_remaining=self._get_players_num_cards_remaining(),
                game_state=self.state,
            )
            self.pass_count = 0
            self.state = CRB
            self.last_domino_played = card
            return evt

        if self.state == DSP:
            # DoubleSixPose
            self.board.card_posed = False
            self.board.left_board = []
            self.board.right_board = []
            self.pose_mode_double_six = True
            self.shuffle_and_distribute()
            self.player_turn = self._find_double_six_player()
            evt = GameEvent(
                event_type=PLAYER_TURN, player=self.player_turn,
                board_state=Board(), player_wins=list(self.player_wins),
                pose_mode_double_six=self.pose_mode_double_six,
                player_cards_remaining=self._get_players_num_cards_remaining(),
                game_state=self.state,
            )
            self.state = EPPDS
            self.last_domino_played = DOUBLE_SIX
            return evt

        if self.state == EPPDS:
            # ExecutePlayerPoseDS
            if self.player_turn != -1:
                card = self.players[self.player_turn].pose_double_six()
                if card != DOUBLE_SIX:
                    return GameEvent(event_type=NULL_EVENT)
                self.update_board(card, POSED)
                self._remove_card_from_player(card, self.player_turn)
                self.last_domino_played = card
            evt = GameEvent(
                event_type=POSED_CARD, player=self.player_turn,
                card=self.board.posed_card, side=POSED,
                board_state=Board(), player_wins=list(self.player_wins),
                pose_mode_double_six=self.pose_mode_double_six,
                player_cards_remaining=self._get_players_num_cards_remaining(),
                game_state=self.state,
            )
            self.pass_count = 0
            self.state = CRB
            return evt

        if self.state == EPT:
            # ExecutePlayerTurn
            player_hands = [list(self.players[i].get_hand()) for i in range(4)]
            evt = GameEvent(
                event_type=PLAYER_TURN, board_state=self.board.copy(),
                player=self.player_turn, player_wins=list(self.player_wins),
                pose_mode_double_six=self.pose_mode_double_six,
                game_state=self.state, player_hands=player_hands,
            )
            if not self._can_play_card(self.player_turn):
                evt.event_type = PASSED
                self.pass_count += 1
                if self.pass_count >= 8:
                    evt.event_type = ROUND_INVALID
                evt.player = self.player_turn
                evt.player_cards_remaining = self._get_players_num_cards_remaining()
            else:
                evt.player_cards_remaining = self._get_players_num_cards_remaining()
                card, side = self.players[self.player_turn].play_card(evt)
                if card == EMPTY_DOMINO or not self._player_has_card(self.player_turn, card):
                    return GameEvent(
                        event_type=NULL_EVENT, player_wins=list(self.player_wins),
                        player_cards_remaining=self._get_players_num_cards_remaining(),
                    )
                # Validate card compatibility
                s1, s2 = get_suits(card)
                card_compatible = False
                if s1 == self.board.left_suit or s1 == self.board.right_suit:
                    card_compatible = True
                if s2 == self.board.left_suit or s2 == self.board.right_suit:
                    card_compatible = True

                card_present = False
                if card in self.board.left_board:
                    card_present = True
                if card in self.board.right_board:
                    card_present = True
                if self.board.card_posed and self.board.posed_card == card:
                    card_present = True

                if not card_compatible or card_present:
                    return GameEvent(
                        event_type=NULL_EVENT, player_wins=list(self.player_wins),
                        player_cards_remaining=self._get_players_num_cards_remaining(),
                    )
                evt.event_type = PLAYED_CARD
                self.pass_count = 0
                evt.player = self.player_turn
                evt.card = card
                evt.side = side
                self.update_board(card, side)
                self._remove_card_from_player(card, self.player_turn)
                evt.player_cards_remaining = self._get_players_num_cards_remaining()
                self.last_domino_played = card
            self.state = CRB
            return evt

        if self.state == CRB:
            # CheckRoundBlock
            evt = GameEvent(
                event_type=NULL_EVENT, player=self.player_turn,
                board_state=self.board.copy(), player_wins=list(self.player_wins),
                pose_mode_double_six=self.pose_mode_double_six,
                player_cards_remaining=self._get_players_num_cards_remaining(),
                game_state=self.state,
            )
            if self._is_game_blocked():
                evt.event_type = ROUND_BLOCKED
            self.state = CRW
            return evt

        if self.state == CRW:
            # CheckRoundWinner
            evt = GameEvent(
                event_type=NULL_EVENT, player=self.player_turn,
                board_state=self.board.copy(), player_wins=list(self.player_wins),
                pose_mode_double_six=self.pose_mode_double_six,
                player_cards_remaining=self._get_players_num_cards_remaining(),
                game_state=self.state,
            )
            # Check if any player has 0 cards
            for i in range(4):
                if self._get_player_num_cards_remaining(i) <= 0:
                    key = 0
                    if self._won_key(self.last_domino_played):
                        key = 1
                        evt.won_key = True
                    self.state = GWC
                    self.last_winner = i
                    self.player_wins[i] += 1
                    self.player_wins[i] += key
                    if self.game_type == "partner":
                        partner = (i + 2) % 4
                        self.player_wins[partner] += 1
                        self.player_wins[partner] += key
                    self.num_rounds += 1
                    evt.player_wins = list(self.player_wins)
                    evt.event_type = ROUND_WIN
                    return evt

            # Check if blocked
            if self._is_game_blocked():
                lowest_player = 0
                lowest_count = self._get_player_hand_count(0)
                drawed_game = False
                for i in range(1, 4):
                    hand_count = self._get_player_hand_count(i)
                    if hand_count < lowest_count:
                        lowest_count = hand_count
                        lowest_player = i
                        drawed_game = False
                    elif hand_count == lowest_count:
                        partner = (lowest_player + 2) % 4
                        if self.game_type == "partner" and partner != i:
                            drawed_game = True
                        elif self.game_type != "partner":
                            drawed_game = True

                if not drawed_game:
                    evt.event_type = ROUND_WIN
                    evt.player = lowest_player
                    self.state = GWC
                    self.last_winner = lowest_player
                    self.player_wins[lowest_player] += 1
                    if self.game_type == "partner":
                        partner = (lowest_player + 2) % 4
                        self.player_wins[partner] += 1
                    self.num_rounds += 1
                    return evt
                else:
                    evt.event_type = ROUND_DRAW
                    self.state = DSP
                    self.num_rounds += 1
                    return evt

            self._advance_player_turn()
            evt.event_type = PLAYER_TURN
            evt.player = self.player_turn
            self.state = EPT
            return evt

        if self.state == GWC:
            # GameWinCheck
            evt = GameEvent(
                event_type=NULL_EVENT, player=self.player_turn,
                board_state=self.board.copy(), player_wins=list(self.player_wins),
                pose_mode_double_six=self.pose_mode_double_six,
                player_cards_remaining=self._get_players_num_cards_remaining(),
                game_state=self.state,
            )
            for i, wins in enumerate(self.player_wins):
                if wins >= 6:
                    self.state = IDLE
                    evt.event_type = GAME_WIN
                    evt.player = i
                    return evt
            self.state = WP
            return evt

        # Fallback
        return GameEvent(
            event_type=NULL_EVENT, player_wins=list(self.player_wins),
            player_cards_remaining=self._get_players_num_cards_remaining(),
            game_state=self.state,
        )
