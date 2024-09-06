import pygame
from Physics import Physics
from Config import BulletSettings

class Bullet(pygame.sprite.Sprite):
    def __init__(self, position, direction):
        super().__init__()
        self.name = "Bullet"
        self.position = pygame.Vector2(position)
        self.velocity = pygame.Vector2(0, 0)
        self.mass = BulletSettings.MASS
        self.on_ground = False
        self.time_alive = 0
        
        # Set up the image and rect
        self.image = pygame.Surface((BulletSettings.RADIUS * 2, BulletSettings.RADIUS * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, pygame.Color('purple'), (BulletSettings.RADIUS, BulletSettings.RADIUS), BulletSettings.RADIUS)
        self.rect = self.image.get_rect(center=self.position)

        # Initialize physics and apply initial force
        self.bullet_physics = Physics()
        self._apply_initial_force(direction)

    def _apply_initial_force(self, direction):
        force_vector = pygame.Vector2(1, 0).rotate(direction) * BulletSettings.FORCE
        self.bullet_physics.apply_force(force_vector, self)

    def update(self, dt_tick):
        self.time_alive += dt_tick * 1000  # Convert dt_tick to milliseconds

        if self.time_alive > BulletSettings.LIFE_TIME:
            self.kill()  # Remove from all groups
        
        # Update physics and position
        self.bullet_physics.update(self, dt_tick)
        self.position += self.velocity * dt_tick
        self.rect.center = self.position
