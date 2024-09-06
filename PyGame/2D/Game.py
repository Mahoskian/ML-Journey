import pygame
from Map import Map
from Player import Player
from Config import GameSettings, MapSettings, BulletSettings, PlayerSettings

# --- Game class ---
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((GameSettings.SCREEN_WIDTH, GameSettings.SCREEN_HEIGHT))
        pygame.display.set_caption("Hammy's Game")
        self.clock = pygame.time.Clock()
        self.running = True
        
        self.dt_tick = 0
        self.tt_tick = 0
        self.frame_count = 0
        
        MapSettings.MAP_OBJ = Map()
        
        PlayerSettings.PLAYER_OBJ = Player()
        PlayerSettings.PLAYER_SPRITE.add(PlayerSettings.PLAYER_OBJ)

    def run(self):
        while self.running:
            self.dt_tick = (self.clock.tick(60) / 1000) * GameSettings.TIME_SCALE
            self.tt_tick += self.dt_tick
            self.frame_count += 1
            self.update(self.dt_tick)
            self.draw()
            self.handle_events()
        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self, dt_tick):
        for sprite in PlayerSettings.PLAYER_SPRITE:
            sprite.update(dt_tick)

    def draw(self):
        self.screen.fill("BLACK")
        # Draw the map first
        MapSettings.MAP_SPRITE.draw(self.screen)
        BulletSettings.BULLET_SPRITE.draw(self.screen)
        # Draw the player on top of the map
        PlayerSettings.PLAYER_SPRITE.draw(self.screen)
        # Render debug information
        self.render_text(f"Frame: {self.frame_count}", (10, 40))
        self.render_text(f"Tick Time: {self.dt_tick:.3f}", (10, 60))
        self.render_text(f"Total Time: {self.tt_tick:.3f}", (10, 80))
        self.render_text(f"(X) Velocity: {PlayerSettings.PLAYER_OBJ.velocity.x:.3f}", (10, 100))
        self.render_text(f"(Y) Velocity: {PlayerSettings.PLAYER_OBJ.velocity.y:.3f}", (10, 120))
        self.render_text(f"(X) Position: {PlayerSettings.PLAYER_OBJ.position.x:.3f}", (10, 140))
        self.render_text(f"(Y) Position: {PlayerSettings.PLAYER_OBJ.position.y:.3f}", (10, 160))
        self.render_text(f"On Ground: {PlayerSettings.PLAYER_OBJ.on_ground}", (10, 180))
        pygame.display.flip()

    def render_text(self, text, position):
        text_surface = pygame.font.Font(None, 18).render(text, True, "GREEN")
        self.screen.blit(text_surface, position)
