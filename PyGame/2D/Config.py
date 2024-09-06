# CONSTANTS FOR EACH CLASS
import pygame

class GameSettings:
    MAP_FILE = 'map_data_01.txt'
    
    SCREEN_WIDTH = 720
    SCREEN_HEIGHT = 720
    SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)
    
    TIME_SCALE = 1

class PlayerSettings:
    PLAYER_OBJ = None
    PLAYER_SPRITE = pygame.sprite.Group()
    SIZE_X = 20
    SIZE_Y = 20
    SIZE = (SIZE_X, SIZE_Y)
    MASS = 1
    
    SPACE_FORCE = 100
    
    W_FORCE = 25
    A_FORCE = 25
    S_FORCE = 25
    D_FORCE = 25
    
    SHOOT_COOLDOWN = 250
    
class PhysicsSettings:
    TERMINAL_VELOCITY = 500
    GRAVITY_ACC = 98*2
    AIR_COEF_FRIC = 0.2
    GROUND_COEF_FRIC = 2

class MapSettings:
    MAP_OBJ = None
    MAP_DATA = None
    
    MAP_SPRITE = pygame.sprite.Group()
    
    MAP_WIDTH = None
    MAP_HEIGHT = None
    
    BLOCK_WIDTH = None
    BLOCK_HEIGHT = None
    
    BLOCK_COLORS = {
            0: "BLACK", # AIR
            1: "WHITE", # WALL
            2: "GREEN", # BOUNCY
            3: "BLUE",  # STICKY
        }
    
class BulletSettings:
    BULLET_SPRITE = pygame.sprite.Group()
    MASS = 1
    FORCE = 500
    RADIUS = 5
    LIFE_TIME = 5000