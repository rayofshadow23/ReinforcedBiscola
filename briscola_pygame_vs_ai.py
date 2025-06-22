import pygame
import sys
import random
from stable_baselines3 import PPO
from briscola_env import BriscolaEnv, decode_id

# --- Costanti grafiche ---
CARD_WIDTH, CARD_HEIGHT = 80, 120
SCREEN_WIDTH, SCREEN_HEIGHT = 900, 600
FPS = 30

# --- Inizializza Pygame ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Briscola - Gioca contro l'AI")
font = pygame.font.SysFont("arial", 24)
clock = pygame.time.Clock()

# --- Colori ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)

# --- Funzioni di utilità ---
def draw_card(surface, x, y, card_text):
    pygame.draw.rect(surface, WHITE, (x, y, CARD_WIDTH, CARD_HEIGHT))
    pygame.draw.rect(surface, BLACK, (x, y, CARD_WIDTH, CARD_HEIGHT), 2)
    text = font.render(card_text, True, BLACK)
    surface.blit(text, (x + 10, y + 50))

def wait_for_card_click(card_positions):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                for idx, (cx, cy) in enumerate(card_positions):
                    if cx <= x <= cx + CARD_WIDTH and cy <= y <= cy + CARD_HEIGHT:
                        return idx
        clock.tick(FPS)

# --- Carica modello e ambiente ---
env = BriscolaEnv()
model = PPO.load("briscola_agent", env=env)
obs, _ = env.reset()
done = False

# --- Main loop ---
while not done:
    screen.fill(GREEN)

    current_player = env.current_player
    screen.blit(font.render(f"Briscola: {env.briscola} (carta: {env.visible_briscola})", True, WHITE), (20, 20))
    screen.blit(font.render(f"Punteggio: TU {env.scores[1]} - BOT {env.scores[0]}", True, WHITE), (20, 60))

    if current_player == 0:
        # BOT
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
    else:
        # UMANO
        screen.blit(font.render("Tocca a te! Clicca su una carta.", True, WHITE), (20, 500))

        hand = obs['hand']
        card_positions = []
        for i, cid in enumerate(hand):
            if cid >= 0:
                card = decode_id(cid)
                x = 200 + i * (CARD_WIDTH + 20)
                y = 350
                draw_card(screen, x, y, card)
                card_positions.append((x, y))

        pygame.display.flip()
        action = wait_for_card_click(card_positions)
        obs, _, done, _, _ = env.step(action)

    # Mostra carte giocate
    played = obs['played']
    if played[0] >= 0:
        draw_card(screen, 200, 200, decode_id(played[0]))
    if played[1] >= 0:
        draw_card(screen, 400, 200, decode_id(played[1]))

    pygame.display.flip()
    clock.tick(FPS)

# --- Fine partita ---
screen.fill(GREEN)
result = "Pareggio!"
if env.scores[0] > env.scores[1]:
    result = "Il bot ha vinto."
elif env.scores[1] > env.scores[0]:
    result = "Hai vinto tu!"
screen.blit(font.render(result, True, WHITE), (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2))
pygame.display.flip()
pygame.time.wait(5000)
pygame.quit()
