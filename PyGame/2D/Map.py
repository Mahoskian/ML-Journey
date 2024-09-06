import pygame
import os
import sys
from Config import GameSettings, MapSettings

# --- Map class ---
class Map(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.load_map()
        self.build_map_sprite()
    
    def resource_path(self, relative_path):
        """Return the absolute path to the resource."""
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)
    
    def load_map(self):
        """Load the map data from the file and initialize map settings."""
        file_path = self.resource_path(GameSettings.MAP_FILE)
        with open(file_path, 'r') as file:
            MapSettings.MAP_DATA = [list(map(int, line.strip().split())) for line in file]
            MapSettings.MAP_WIDTH = len(MapSettings.MAP_DATA[0])
            MapSettings.MAP_HEIGHT = len(MapSettings.MAP_DATA)
            MapSettings.BLOCK_WIDTH = GameSettings.SCREEN_WIDTH // MapSettings.MAP_WIDTH
            MapSettings.BLOCK_HEIGHT = GameSettings.SCREEN_HEIGHT // MapSettings.MAP_HEIGHT
    
    def build_map_sprite(self):
        """Create sprites for each tile in the map and add them to the map sprite group."""
        tile_size = min(MapSettings.BLOCK_WIDTH, MapSettings.BLOCK_HEIGHT)
        tile_class_mapping = {
            1: Wall,
            2: BouncyBlock,
            3: StickyBlock
        }
        for y, row in enumerate(MapSettings.MAP_DATA):
            for x, tile in enumerate(row):
                TileClass = tile_class_mapping.get(tile)
                if TileClass:
                    tile_sprite = TileClass(x * tile_size, y * tile_size, tile_size, tile_size)
                    MapSettings.MAP_SPRITE.add(tile_sprite)

# --- Block Classes ---
class Wall(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height):
        super().__init__()
        self.image = pygame.Surface((width, height))
        self.image.fill("PINK")
        self.rect = self.image.get_rect(topleft=(x, y))

class BouncyBlock(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height):
        super().__init__()
        self.image = pygame.Surface((width, height))
        self.image.fill("GREEN")
        self.rect = self.image.get_rect(topleft=(x, y))

class StickyBlock(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height):
        super().__init__()
        self.image = pygame.Surface((width, height))
        self.image.fill("BLUE")
        self.rect = self.image.get_rect(topleft=(x, y))
