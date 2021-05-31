# -*- coding: utf-8 -*-
"""
Created on Sun May 30 22:23:36 2021

@author: Christoffer
"""

import pygame
import sys
import random
from pygame.math import Vector2
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import time


cell_size = 40
cell_number = 10


class SNAKE:
    def __init__(self):
        self.body = [Vector2(2,5), Vector2(1,5), Vector2(0,5)]
        self.direction = Vector2(0,0)
        self.new_block = False
        self.alive = True
        self.score = (len(self.body) - 3)
        #Images for snakes different positions.
        self.head_up = pygame.image.load('Graphics/head_up.png').convert_alpha()
        self.head_down = pygame.image.load('Graphics/head_down.png').convert_alpha()
        self.head_right = pygame.image.load('Graphics/head_right.png').convert_alpha()
        self.head_left = pygame.image.load('Graphics/head_left.png').convert_alpha()
        self.tail_up = pygame.image.load('Graphics/tail_up.png').convert_alpha()
        self.tail_down = pygame.image.load('Graphics/tail_down.png').convert_alpha()
        self.tail_right = pygame.image.load('Graphics/tail_right.png').convert_alpha()
        self.tail_left = pygame.image.load('Graphics/tail_left.png').convert_alpha()
        self.body_vertical = pygame.image.load('Graphics/body_vertical.png').convert_alpha()
        self.body_horizontal = pygame.image.load('Graphics/body_horizontal.png').convert_alpha()
        self.body_tr = pygame.image.load('Graphics/body_tr.png').convert_alpha()
        self.body_tl = pygame.image.load('Graphics/body_tl.png').convert_alpha()
        self.body_br = pygame.image.load('Graphics/body_br.png').convert_alpha()
        self.body_bl = pygame.image.load('Graphics/body_bl.png').convert_alpha()
        
    def draw_snake(self):
        self.update_head_graphics()
        self.update_tail_graphics()
        
        for index,block in enumerate(self.body):
            x_pos = int(block.x * cell_size)
            y_pos = int(block.y * cell_size)
            block_rect = pygame.Rect(x_pos,y_pos,cell_size,cell_size)
            
            if index == 0:
                screen.blit(self.head,block_rect)
            elif index == len(self.body) - 1:
                screen.blit(self.tail,block_rect)
            else:
                previous_block = self.body[index + 1] - block
                next_block= self.body[index - 1] - block
                if previous_block.x == next_block.x:
                    screen.blit(self.body_vertical,block_rect)
                elif previous_block.y == next_block.y:
                    screen.blit(self.body_horizontal,block_rect)
                else:
                    if previous_block.x == -1 and next_block.y == -1 or previous_block.y == -1 and next_block.x == -1:
                        screen.blit(self.body_tl,block_rect)
                    elif previous_block.x == -1 and next_block.y == 1 or previous_block.y == 1 and next_block.x == -1:
                        screen.blit(self.body_bl,block_rect)
                    elif previous_block.x == 1 and next_block.y == -1 or previous_block.y == -1 and next_block.x == 1:
                        screen.blit(self.body_tr,block_rect)
                    elif previous_block.x == 1 and next_block.y == 1 or previous_block.y == 1 and next_block.x == 1:
                        screen.blit(self.body_br,block_rect)

            
    def update_head_graphics(self):
        head_relation = self.body[1] - self.body[0] #New vector of past position vs current, gives direction
        if head_relation == Vector2(1,0):
            self.head = self.head_left
        elif head_relation == Vector2(-1,0):
            self.head = self.head_right
        elif head_relation == Vector2(0,1):
            self.head = self.head_up
        elif head_relation == Vector2(0,-1):
            self.head = self.head_down
            
    def update_tail_graphics(self):
        tail_relation = self.body[-2] - self.body[-1]
        if tail_relation == Vector2(1,0):
            self.tail = self.tail_left
        elif tail_relation == Vector2(-1,0):
            self.tail = self.tail_right
        elif tail_relation == Vector2(0,1):
            self.tail = self.tail_up
        elif tail_relation == Vector2(0,-1):
            self.tail = self.tail_down        
            
    def move_snake(self):
        if self.new_block == True:
            body_copy = self.body[:]
            body_copy.insert(0,body_copy[0] + self.direction)
            self.body = body_copy[:]
            self.new_block = False
        else:
            body_copy = self.body[:-1]
            body_copy.insert(0,body_copy[0] + self.direction)
            self.body = body_copy[:]
        
        
    def add_block(self):
        self.new_block = True

    def reset(self):
        self.score = (len(self.body) - 3)
        self.body = [Vector2(2,5), Vector2(1,5), Vector2(0,5)]
        self.direction = Vector2(0,0)
        
        
class DONUT:
    def __init__(self):
        self.randomize()
    
    def draw_donut(self):
        donut_rect = pygame.Rect(int(self.pos.x * cell_size), int(self.pos.y * cell_size), cell_size, cell_size)
        screen.blit(donut, donut_rect)
        
    def randomize(self):
        self.x = random.randint(0, cell_number -1)
        self.y = random.randint(0, cell_number -1)
        self.pos = Vector2(self.x,self.y)
        
class MAIN:
    def __init__(self):
        self.snake = SNAKE()
        self.donut = DONUT()
        
    def update(self):
        self.snake.move_snake()
        self.check_collision()
        self.check_fail()
        
    def draw_elements(self):
        self.draw_grass()
        self.donut.draw_donut()
        self.snake.draw_snake()
        self.draw_score()
        
    def check_collision(self):
        if self.donut.pos == self.snake.body[0]:
            self.donut.randomize()
            self.snake.add_block()
            
        for block in self.snake.body[1:]:
            if block == self.donut.pos:
                self.donut.randomize()
            
    def check_fail(self):
        #If snakes head is not in cell between 0 and cell_number -1
        if not 0 <= self.snake.body[0].x < cell_number or not 0 <= self.snake.body[0].y < cell_number:
            self.game_over()
            
        for block in self.snake.body[1:]:
            if block == self.snake.body[0]:
                self.game_over()
            
    def game_over(self):
        self.snake.alive = False
        self.snake.reset()
            
    def draw_grass(self):
        grass_color = (167,209,61)
        for row in range(cell_number):
            if row % 2 == 0:
                for col in range(cell_number):
                    if col % 2 == 0:
                        grass_rect = pygame.Rect(col * cell_size,row * cell_size,cell_size,cell_size)
                        pygame.draw.rect(screen, grass_color, grass_rect)
            else:
                for col in range(cell_number):
                    if col % 2 != 0:
                        grass_rect = pygame.Rect(col * cell_size,row * cell_size,cell_size,cell_size)
                        pygame.draw.rect(screen, grass_color, grass_rect)
                        
    def draw_score(self):
        score_text = str(len(self.snake.body) - 3)
        score_surface = game_font.render(score_text, True, (56,74,12))
        score_x = int(cell_size * cell_number - 60)
        score_y = int(cell_size * cell_number - 40)
        score_rect = score_surface.get_rect(center = (score_x, score_y))
        donut_rect = donut.get_rect(midright = (score_rect.left, score_rect.centery))
        
        screen.blit(score_surface, score_rect)    
        screen.blit(donut, donut_rect)
        
    def traverseObservation(self, x_steps, y_steps, x_head, y_head, intercardinal = False):
        x_head = x_head + 1
        y_head = y_head + 1
        steps = 1
        donut = -1
        body = -1
        boundary = -1

        x = x_head + x_steps
        y = y_head + y_steps
        max_x = cell_number
        max_y = cell_number

        while (x > -1) and (y > -1) and (x < max_x) and (y < max_y):
            if self.donut.pos.x == x and self.donut.pos.y == y:
                if donut == -1: 
                    if intercardinal == True:
                        donut = steps * 2
                    else:
                        donut = steps
                    #print("Observed donut at:", steps, "steps")
                    
            #if not 0 <= x < cell_number or not 0 <= y < cell_number:
            if x >= cell_number -1 or y >= cell_number -1 or x < 1 or y < 1:
                if boundary == -1:
                    if intercardinal == True:
                        boundary = steps * 2
                    else:
                        boundary = steps
                    #print("Observed boundary at:", steps, "steps")

            for block in self.snake.body[1:]:
                if block.x == x and block.y == y:
                    if body == -1:
                        if intercardinal == True:
                            body = steps * 2
                        else:
                            body = steps
                    #print("Observed body at:", steps, "steps")
                            
            steps += 1
            x += x_steps
            y += y_steps

        return [donut,boundary,body]
        
    def observe(self):
        x = int(self.snake.body[0].x)
        y = int(self.snake.body[0].y)

        observation = np.array([                
                    # up
                    self.traverseObservation(0, -1, x, y),
                    # up right
                    self.traverseObservation(1, -1, x, y, intercardinal=True),
                    # right
                    self.traverseObservation(1, 0, x, y),
                    # down right
                    self.traverseObservation(1, 1, x, y, intercardinal=True),
                    # down
                    self.traverseObservation(0, 1, x, y),
                    # down left
                    self.traverseObservation(-1, 1, x, y, intercardinal=True),
                    # left
                    self.traverseObservation(-1, 0, x, y),
                    # up left
                    self.traverseObservation(-1, -1, x, y, intercardinal=True),])

        return observation
        
    def think(self, observations):
        best_action = np.argmax(observations)
        if best_action == 0:
            return (0,-1)
        if best_action == 1:
            return (0, 1)
        if best_action == 2:
            return (1, 0)
        if best_action == 3:
            return (-1, 0)
        
        
def generate_network():
    i = Input(shape=[8, 3])
    network = Dense(24)(i)
    network = Dense(24)(network)
    network = Dense(4, activation='softmax')(network)
    model = Model(i, network)
    return model

def compile_model(model):
    model.compile()
    return model

def fitness_function(donuts, maxscore, minscore = 0):
    return (donuts-minscore)/(maxscore-minscore)

def generate_population(population_size):
    population = []
    for n in range(population_size):
        model = generate_network()
        model = compile_model(model)
        population.append(model)
    return population

def mutate_organism(organism, my):
    weights = organism.get_weights()
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            if weights[i][j].any() == 0.0:
                continue
            for x in range(len(weights[i][j])):
                if np.random.random()<my:
                    weights[i][j][x] += round(random.uniform(-1, 1), 8)
    offspring = generate_network()
    offspring.set_weights(weights)
    compile_model(offspring)     
    return offspring

population_size = 100
population = generate_population(population_size)

framerate = 60
ms_timer = 10
generations = 250
#my=0.1
max_donuts = (cell_number * cell_number)-4
mean_fitness = []

pygame.init()
pygame.display.set_caption('Agent SNEK')
programIcon = pygame.image.load('graphics/donut40x40.png')
pygame.display.set_icon(programIcon)
screen = pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))
clock = pygame.time.Clock()
donut = pygame.image.load('graphics/donut40x40.png').convert_alpha()
game_font = pygame.font.Font(None, 35)

main_game = MAIN()

SCREEN_UPDATE = pygame.USEREVENT
pygame.time.set_timer(SCREEN_UPDATE, ms_timer)

while True:
    for generation in range(generations):
        my = 0.1 - (((generation+1)*0.0001)*3.6)
        start_time = time.time()
        scores = []
        for organism in population:
            max_steps = (main_game.snake.score+1)*50
            steps = 0
            main_game.snake.alive = True
            main_game.donut.randomize()
            while main_game.snake.alive == True:
                for event in pygame.event.get():
                    if main_game.snake.alive == False:
                        break
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()   
                    if event.type == SCREEN_UPDATE:
                        observations = main_game.observe()
                        action_potential = organism(observations.reshape(1,8,3))[0,0]
                        action = main_game.think(action_potential)
                        if action == (0,-1): #UP
                            if main_game.snake.direction.y != 1:
                                main_game.snake.direction = Vector2(action)
                        if action == (0, 1): #DOWN
                            if main_game.snake.direction.y != -1:
                                main_game.snake.direction = Vector2(action)
                        if action == (1, 0): #RIGHT
                            if main_game.snake.direction.x != -1:
                                main_game.snake.direction = Vector2(action)
                        if action == (-1, 0): #LEFT
                            if main_game.snake.direction.x != 1:
                                main_game.snake.direction = Vector2(action)
                        steps += 1
                        main_game.update()
                        if steps >= max_steps:
                            scores.append(fitness_function(main_game.snake.score, max_donuts))
                            main_game.snake.alive = False
                            main_game.snake.reset()
                            break
                        if main_game.snake.alive == False:
                            scores.append(fitness_function(main_game.snake.score, max_donuts))
                            break
                    screen.fill((175,215,70))
                    main_game.draw_elements()
                    pygame.display.update()
                    clock.tick(framerate)
    
    
        maxscore = max(scores)
        next_population = []
        for n in range(population_size):
            rand_organism = np.random.randint(population_size)
            if n<3:
                rand_organism = scores.index(maxscore)
            else:
                rand_organism_2 = np.random.randint(population_size)
                if scores[rand_organism_2]>scores[rand_organism]:
                    rand_organism=rand_organism_2
            next_population.append(mutate_organism(population[rand_organism], my))
        population = next_population
        end_time = time.time()
        mean_fitness.append(sum(scores)/len(scores))
        print("Generation:",generation+1, ", Max fitness:",maxscore,", Max donuts:",round(maxscore*(cell_number*cell_number)),
              ", Average fitness:",sum(scores)/len(scores),
              ", Time passed(minutes):", round(end_time - start_time)/60)
        
    
    champion = scores.index(maxscore)
    champion_model = population[champion]
    champion_model.save('/saved_model/champion_organism_250.h5')
    
    orgint = 1
    for organism in population:
        filename = 'saved_models/'+str(orgint)+'_organism_250.h5' 
        organism.save(filename)
        orgint += 1
    
    textfile = open("250_mean_fitness.txt", "w")
    for element in mean_fitness:
        textfile.write(str(element) + "\n")
    textfile.close()       
        
    
    pygame.quit()
    #sys.exit()  

