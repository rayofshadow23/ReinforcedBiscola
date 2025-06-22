import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random

# Valore delle carte
CARD_VALUES = {
    'A': 11,
    '3': 10,
    'K': 4,
    'Q': 3,
    'J': 2,
    '7': 0,
    '6': 0,
    '5': 0,
    '4': 0,
    '2': 0
}

# Ordine di presa
CARD_ORDER = ['2', '4', '5', '6', '7', 'J', 'Q', 'K', '3', 'A']

# Semi
SUITS = ['C', 'D', 'H', 'S']  # Coppe, Denari, Cuori, Spade

def card_id(card):
    """Map card (e.g. 'AC') to unique ID 0-39"""
    value, suit = card[:-1], card[-1]
    return CARD_ORDER.index(value) + 10 * SUITS.index(suit)

def decode_id(cid):
    """Map card ID back to string"""
    return CARD_ORDER[cid % 10] + SUITS[cid // 10]

def compare_cards(c1, c2, briscola):
    """Return winner: 0 if c1 wins, 1 if c2 wins"""
    v1, s1 = c1[:-1], c1[-1]
    v2, s2 = c2[:-1], c2[-1]

    if s1 == s2:
        return 0 if CARD_ORDER.index(v1) > CARD_ORDER.index(v2) else 1
    elif s1 == briscola and s2 != briscola:
        return 0
    elif s2 == briscola and s1 != briscola:
        return 1
    else:
        return 0  # chi ha messo la prima carta prende

class BriscolaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.deck = []
        self.hands = [[], []]
        self.briscola = ''
        self.visible_briscola = ''
        self.played_cards = []
        self.scores = [0, 0]
        self.current_player = 0
        self.max_turns = 20  # 20 mani

        self.observation_space = spaces.Dict({
            "hand": spaces.MultiDiscrete([41] * 3),
            "played": spaces.MultiDiscrete([41, 41]),
            "briscola": spaces.Discrete(4),
            "scores": spaces.Box(low=0, high=120, shape=(2,), dtype=np.int32),
            "remaining_cards": spaces.Discrete(41),
        })

        self.action_space = spaces.Discrete(3)  # Gioca una delle 3 carte in mano

    def reset(self, seed=None, options=None):
        self.deck = [v + s for v in CARD_ORDER for s in SUITS]
        random.shuffle(self.deck)

        self.visible_briscola = self.deck.pop()
        self.briscola = self.visible_briscola[-1]

        self.hands = [[self.deck.pop() for _ in range(3)] for _ in range(2)]
        self.played_cards = []
        self.scores = [0, 0]
        self.current_player = 0
        self.turn = 0

        return self._get_obs(), {}

    def _get_obs(self):
        hand = self.hands[self.current_player]
        # Padding con 40 per avere sempre 3 elementi
        hand_ids = [card_id(c) if c is not None else 40 for c in hand]
        while len(hand_ids) < 3:
            hand_ids.append(40)

        played_ids = [card_id(c) if c is not None else 40 for c in self.played_cards]
        while len(played_ids) < 2:
            played_ids.append(40)

        obs = {
            "hand": np.array(hand_ids, dtype=np.int32),
            "played": np.array(played_ids, dtype=np.int32),
            "briscola": SUITS.index(self.briscola),
            "scores": np.array(self.scores, dtype=np.int32),
            "remaining_cards": np.int32(len(self.deck))
            }
        return obs




    def step(self, action):
        hand = self.hands[self.current_player]
	
		# ✅ Se la mano è vuota, termina immediatamente la partita
        if len(hand) == 0:
            return self._get_obs(), 0, True, False, {"error": "Empty hand"}
	
		# ✅ Azione corretta se fuori range
        if action < 0 or action >= len(hand):
            action = 0  # oppure random.choice(range(len(hand)))
	
        card = hand.pop(action)
        self.played_cards.append(card)
	
        done = False
        reward = 0
	
        if len(self.played_cards) == 2:
            winner = compare_cards(*self.played_cards, self.briscola)
            points = sum(CARD_VALUES[c[:-1]] for c in self.played_cards)
	
            self.scores[winner] += points
            self.turn += 1
	
            first = winner
			# ✅ Pesca sicura
            if len(self.deck) >= 2:
                self.hands[first].append(self.deck.pop())
                self.hands[1 - first].append(self.deck.pop())
            elif len(self.deck) == 1:
                self.hands[first].append(self.deck.pop())
			# else: non si pesca più
	
            self.played_cards = []
            self.current_player = winner
            reward = points if self.current_player == 0 else -points
        else:
            self.current_player = 1 - self.current_player
	
		# ✅ Se una mano è vuota prematuramente, chiudiamo la partita
        if len(self.hands[0]) == 0 and len(self.hands[1]) == 0 and len(self.deck) == 0:
            done = True
	
        if self.turn >= self.max_turns:
            done = True
            if self.scores[0] > self.scores[1]:
                reward += 100
            elif self.scores[0] < self.scores[1]:
                reward -= 100
	
        return self._get_obs(), reward, done, False, {}

    def render(self):
        print(f"Player {self.current_player} hand: {self.hands[self.current_player]}")
        print(f"Played cards: {self.played_cards}")
        print(f"Scores: {self.scores}")
        print(f"Briscola: {self.briscola}")

