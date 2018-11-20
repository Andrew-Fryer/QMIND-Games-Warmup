import pygame
import time         # importing libraries
import random
import math

pygame.init()   # initialize pygame
dh = 600
dw = 800        # define room boundaries

black = (0, 0, 0)
white = (255, 255, 255) # defining basic colour codes
red = (255, 0, 0)

gameDisplay = pygame.display.set_mode((dw, dh)) # setting up the game window
pygame.display.set_caption('testing')
clock = pygame.time.Clock()

pacImg = pygame.image.load('pacman.png')        # loading the object sprite
pacwid = 75                                     # dimensions of the pacman object sprite
pachi = 75

def dotsgood(score):                        # funtion for rendering score
    font = pygame.font.SysFont(None, 25)
    text = font.render("Score:" + str(score), True, black)
    gameDisplay.blit(text,(0,0))            # blit the text to the screen

def things(x,y,w,h,c):
    pygame.draw.rect(gameDisplay,c,[x,y,w,h])   # drawing enemy rectangles


def text_objects(text, font):
    textSurf = font.render(text, True, black)       # setting up text displays
    return textSurf, textSurf.get_rect()

def message_display(text):
    largeText = pygame.font.Font('freesansbold.ttf',115)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.center = ((dw/2),(dh/2))       # defining message text and location
    gameDisplay.blit(TextSurf, TextRect)

    pygame.display.update()         # update the display

    time.sleep(2)           # pause the game for 2 seconds

    gameloop()          # restart the game loop

def crash():
    message_display('you died')

def pac(x,y):
    gameDisplay.blit(pacImg,(x,y))      # plot the position of pacman

def gameloop():                         # main loop for playing the game
                            # defining game variables
    clockspeed = 90         # speed of the game (bigger = faster)
    x = (dw-pacwid)/2       # starting position (centre of screen)
    y = (dh-pachi)/2
    speed = 4               # speed of pacman
    objfact = 0.75          # proportional enemy speed
    ldir = 0                # directional flags
    rdir = 0
    udir = 0
    ddir = 0

    addobj = 5                # frequency of adding an enemy (per score)
    objcount = 1            # starting number of enemies
    anglevar = 20           # accuracy hitting pacman
    objw = 50               # dimensions of enemies
    objh = 50
    objspeed = speed * objfact
    incfreq = 1                 # how often the speed increases
    increase = 0                # how much the speed increases each time
    score = 0                   # initial score

    objyi = []
    objxi = []                  # defining arrays to store enemy positions
    objdir = []

    for i in range(0,objcount):
        objyi.append(0)
        objxi.append(0)
        objdir.append(0)

    for i in range(0,objcount):         # defining initial enemy traits
        side = random.randrange(0,2)
        if side == 1:
            horside = random.randrange(-1,2,2)
            horrand = 0
            verside = 0
            verrand = 1
        else :
            horside = 0
            horrand = 1
            verside = random.randrange(-1, 2, 2)
            verrand = 0
        porp = random.randrange(0,100)/100
        objxi[i] = abs(horside*(dw-pacwid)/2)+horside*dw/2+horrand*porp*dw      # start somewhere on the edge of the screen
        objyi[i] = abs(verside*(dh-pachi)/2)+verside*dh/2+verrand*porp*dh
        angle = math.atan((y-objyi[i])/(x-objxi[i]))
        if x-objxi[i] < 0:
            angle = 3.14159265 + angle                                          # aim within the given accuracy of pacman
        angle = int(angle * 180/3.1415926535)
        objdir[i] = random.randrange(angle-anglevar, angle+anglevar)*3.1415926535/180

    crashed = False

    while not crashed:              # gameplay phase

        for event in pygame.event.get():        # checking events occuring on the frame

            if event.type == pygame.QUIT:       # exit condition
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:            # checking keystrokes for direction
                if event.key == pygame.K_LEFT:
                    ldir = -speed
                elif event.key == pygame.K_RIGHT:
                    rdir = speed
                if event.key == pygame.K_DOWN:
                    ddir = speed
                elif event.key == pygame.K_UP:
                    udir = -speed
            if event.type == pygame.KEYUP:              # key release checks
                if event.key == pygame.K_RIGHT:
                    rdir = 0
                elif event.key == pygame.K_DOWN:
                    ddir = 0
                elif event.key == pygame.K_UP:
                    udir = 0
                elif event.key == pygame.K_LEFT:
                    ldir = 0




            print(event)        # print the events of the frame

        x += ldir + rdir        # update pacmans position
        y += udir + ddir
        gameDisplay.fill(white)     # background colour

        for i in range(0,objcount):
            things(objxi[i],objyi[i],objw,objh,red) # draw the enemy positions

            objxi[i] += objspeed*math.cos(objdir[i])  # update enemy positions
            objyi[i] += objspeed*math.sin(objdir[i])

        pac(x,y)            # draw pacmans position
        dotsgood(score)     # draw score



        if x > dw-pacwid/2 or x < -pacwid/2:            #check if pacman off screen
            crash()
        if y > dh-pachi/2 or y < -pachi/2:
            crash()

        for i in range(0, objcount):            # enemy checks

            if objyi[i] < -objh or objyi[i] > dh or objxi[i]<-objw or objxi[i]> dw: # if enemy off screen

                side = random.randrange(0, 2)
                if side == 1:
                    horside = random.randrange(-1, 2, 2)
                    horrand = 0
                    verside = 0
                    verrand = 1
                else:                                                                   # generate new enemy positions
                    horside = 0
                    horrand = 1
                    verside = random.randrange(-1, 2, 2)
                    verrand = 0
                porp = random.randrange(0, 100) / 100
                objxi[i] = abs(horside * (dw - pacwid) / 2) + horside * dw / 2 + horrand * porp * dw
                objyi[i] = abs(verside * (dh - pachi) / 2) + verside * dh / 2 + verrand * porp * dh         # respawn enemy
                angle = math.atan((y - objyi[i]) / (x - objxi[i]))
                if x - objxi[i] < 0:
                    angle = 3.14159265 + angle
                angle = int(angle * 180 / 3.1415926535)
                objdir[i] = random.randrange(angle - anglevar, angle + anglevar) * 3.1415926535 / 180

                if score % incfreq ==0:
                    objspeed += increase        # increase enemy speed

                score += 1                      # increment score

                if score % addobj == 0:         # adding additional enemy

                    objcount += 1               # increment count

                    side = random.randrange(0, 2)
                    if side == 1:
                        horside = random.randrange(-1, 2, 2)
                        horrand = 0
                        verside = 0
                        verrand = 1
                    else:                                                   # generate new position
                        horside = 0
                        horrand = 1
                        verside = random.randrange(-1, 2, 2)
                        verrand = 0
                    porp = random.randrange(0, 100) / 100
                    objxi.append(abs(horside * (dw - pacwid) / 2) + horside * dw / 2 + horrand * porp * dw)     # append position list
                    objyi.append(abs(verside * (dh - pachi) / 2) + verside * dh / 2 + verrand * porp * dh)
                    angle = math.atan((y - objyi[objcount-1]) / (x - objxi[objcount-1]))
                    if x - objxi[objcount-1] < 0:
                        angle = 3.14159265 + angle
                    angle = int(angle * 180 / 3.1415926535)
                    objdir.append(random.randrange(angle - anglevar, angle + anglevar) * 3.1415926535 / 180)       # create new object

            if (y > objyi[i] and y < objyi[i] + objh) or (y + pachi > objyi[i] and y + pachi < objyi[i] + objh) or (y<objyi[i] and y + pachi > objyi[i] + objh):
                if (x > objxi[i] and x < objxi[i] + objw) or (x + pacwid > objxi[i] and x + pacwid < objxi[i] + objw) or (x<objxi[i] and x + pacwid > objxi[i] + objw):
                    crash()
            #^^^^^ crash condition
        pygame.display.update()     # update screen
        clock.tick(clockspeed)      # clock progress

gameloop()      # run the game
pygame.quit()   # quit the game environment
quit()
