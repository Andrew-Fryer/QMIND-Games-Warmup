import copy
import random
import pygame
import sys


class PieceType(object):
    def __init__(self, name, bitMaps):
        self.name = name
        self.bitMaps = bitMaps


def setTrue(bitMap, listOfFilledSquares):
    for row, col in listOfFilledSquares:
        bitMap[row][col] = True
    return bitMap


# constants

def bitMapEmpty(numRows, numCols):
    return [[False for col in range(numCols)] for row in range(numRows)]


# list of all the pieces. Please see PieceType to understand this
pieces = [PieceType("I", [setTrue(bitMapEmpty(4, 1), [(0, 0), (1, 0), (2, 0), (3, 0)]),
                          setTrue(bitMapEmpty(1, 4), [(0, 0), (0, 1), (0, 2), (0, 3)])]),
          PieceType("O", [setTrue(bitMapEmpty(2, 2), [(0, 0), (1, 0), (1, 1), (0, 1)])]),
          PieceType("T", [setTrue(bitMapEmpty(2, 3), [(0, 1), (1, 1), (1, 0), (1, 2)]),
                          setTrue(bitMapEmpty(3, 2), [(0, 1), (1, 0), (1, 1), (2, 1)]),
                          setTrue(bitMapEmpty(2, 3), [(0, 0), (0, 1), (0, 2), (1, 1)]),
                          setTrue(bitMapEmpty(3, 2), [(0, 0), (1, 0), (1, 1), (2, 0)])]),
          PieceType("S", [setTrue(bitMapEmpty(3, 2), [(0, 0), (1, 0), (1, 1), (2, 1)]),
                          setTrue(bitMapEmpty(2, 3), [(0, 0), (0, 1), (1, 1), (1, 2)])]),
          PieceType("Z", [setTrue(bitMapEmpty(3, 2), [(0, 0), (1, 0), (1, 1), (2, 1)]),
                          setTrue(bitMapEmpty(2, 3), [(0, 1), (0, 2), (1, 0), (1, 1)])]),
          PieceType("J", [setTrue(bitMapEmpty(3, 2), [(0, 0), (0, 1), (1, 1), (2, 1)]),
                          setTrue(bitMapEmpty(2, 3), [(0, 0), (0, 1), (0, 2), (1, 0)]),
                          setTrue(bitMapEmpty(3, 2), [(0, 0), (1, 0), (2, 0), (2, 1)]),
                          setTrue(bitMapEmpty(2, 3), [(0, 2), (1, 0), (1, 1), (1, 2)])]),
          PieceType("L", [setTrue(bitMapEmpty(3, 2), [(0, 0), (0, 1), (1, 0), (2, 0)]),
                          setTrue(bitMapEmpty(2, 3), [(0, 0), (1, 0), (1, 1), (1, 2)]),
                          setTrue(bitMapEmpty(3, 2), [(0, 1), (1, 1), (2, 0), (2, 1)]),
                          setTrue(bitMapEmpty(2, 3), [(0, 0), (0, 1), (0, 2), (1, 2)])])]
black = (0, 0, 0)
white = (255, 255, 255)

gridWidth = 10
gridHeight = 40
squareLength = 10  # pixels

class Piece(object):
    def __init__(self, pieceNum):
        self.piece = pieces[pieceNum]
        self.ori = 0
        self.col = 3
        self.row = 36
    def width(self):
        return len(self.piece.bitMaps[self.ori][0])
    def height(self):
        return len(self.piece.bitMaps[self.ori])

class Game(object): # pull our human things like moveLeft

    def __init__(self):  # add a flag for computer or human?
        self.grid = [[False for i in range(gridWidth)] for j in range(gridHeight)]  # [0][0] is bottom-left corner, index row, then col ([x][y] on caresian plane)
        # manually load pieces
        self.currentPiece = Piece(random.randint(0, 6))  # there are 7 pieces to choose from
        self.nextPiece = Piece(random.randint(0, 6))

        # set up a pygame window for this game
        pygame.init()
        self.screen = pygame.display.set_mode((gridWidth * gridWidth, gridHeight * gridWidth))
        self.screen.fill(white)
        pygame.display.update()

    def rotatePiece(self):
        self.currentPiece.ori = (self.currentPiece + 1) % len(self.currentPiece.piece.bitMaps)

    def moveLeft(self):
        # check that there is space
        if self.col < 0:
            return None
        self.currentPiece.col -= 1

    def moveRight(self):
        # check that there is space
        if self.currentPiece.col + self.currentPiece.width() >= 10:
            return None
        self.currentPiece.col += 1

    def isAlive(self):
        # we die if any of the top 4 rows aren't empty
        for row in range(gridHeight - 4, gridHeight):
            for col in range(gridWidth):
                if self.grid[row][col]:
                    # exit pygame
                    game.render()
                    pygame.display.quit()
                    sys.exit()

                    return False
        return True

    def dropPiece(self):

        # for each col in the bitMap, find the number of sqaures the piece can move down
        minNumSquares = 40
        print("currentPiece looks like", self.currentPiece.piece.bitMaps[self.currentPiece.ori])

        for col in range(self.currentPiece.col, self.currentPiece.col + self.currentPiece.width()):  # this won't work when pieces are touching the right side
            # numSquares free in bitMap
            numSquares = 0
            row = 0
            while row < self.currentPiece.height() and not self.currentPiece.piece.bitMaps[self.currentPiece.ori][row][col - self.currentPiece.col]:
                numSquares += 1
                row += 1
            print("numSquares in bitMap at col:", col, "is:", numSquares)
            # numSquares free on board
            i = self.currentPiece.row - 1
            while i >= 0 and not self.grid[i][col]:
                numSquares += 1
                i -= 1
            print("total numSquares at col:", col, "is", numSquares)
            if numSquares < minNumSquares:
                minNumSquares = numSquares
        print("the minimum numSquares is", minNumSquares)
        # advance the piece down the page
        print("currentPiece row was:", self.currentPiece.row)
        self.currentPiece.row -= minNumSquares
        print("currentPiece row is now", self.currentPiece.row)

        # add piece to the grid
        for row in range(self.currentPiece.row, self.currentPiece.row + self.currentPiece.height()):
            for col in range(self.currentPiece.col, self.currentPiece.col + self.currentPiece.width()):
                if row >= gridHeight or col >= gridWidth:
                    print("waaht")
                if self.grid[row][col] and self.currentPiece.piece.bitMaps[self.currentPiece.ori][row - self.currentPiece.row][col - self.currentPiece.col]:
                    print("the bitmap is overlapping on to filled squares")
                self.grid[row][col] = self.grid[row][col] or self.currentPiece.piece.bitMaps[self.currentPiece.ori][row - self.currentPiece.row][col - self.currentPiece.col]  # should never both be True
        # take out any complete rows
        completeRows = []
        for row in range(self.currentPiece.row, self.currentPiece.row + self.currentPiece.height()):
            isComplete = True
            for col in range(10):
                if not self.grid[row][col]:
                    isComplete = False  # continue to for row?  # move on to the next row
            if isComplete:
                completeRows.append(row)
        for row in reversed(completeRows): # reversed so that the indexes don't need to be adjusted for shifting the values of the grid
            # remove row
            print("row:", row, "is complete. removing...")
            for myRow in range(row, gridHeight - 1):
                for myCol in range(gridWidth):
                    self.grid[row][col] = self.grid[row + 1][col]
            for myCol in range(gridWidth):
                self.grid[39][myCol] = False  # do last row seperately

        # load new piece
        self.currentPiece = self.nextPiece
        print("currentPiece is now:", self.currentPiece.piece.name)
        self.nextPiece = Piece(random.randint(0, 6)) # there are 7 pieces
        print("nextPiece is now:", self.nextPiece.piece.name)


    def render(self):
        self.screen.fill(white)
        numFilledSquares = 0
        for row in range(gridHeight):
            for col in range(gridWidth):
                if self.grid[row][col]:
                    pygame.draw.rect(self.screen, black, pygame.Rect(col * squareLength, (gridHeight - row) * squareLength, squareLength, squareLength))
                    numFilledSquares += 1
        print("numFilled Squares in the grid:", numFilledSquares)
        pygame.display.update()
        # draw the current piece in the top left corner
        for row in range(self.currentPiece.height()):
            for col in range(self.currentPiece.width()):
                if self.currentPiece.piece.bitMaps[self.currentPiece.ori][row][col]:
                    pygame.draw.rect(self.screen, black, pygame.Rect(col * squareLength, (self.currentPiece.height() - row) * squareLength, squareLength, squareLength))
        pygame.display.update()
        # draw the next piece in the top right corner
        for row in range(self.nextPiece.height()):
            for col in range(self.nextPiece.width()):
                if self.nextPiece.piece.bitMaps[self.nextPiece.ori][row][col]:
                    pygame.draw.rect(self.screen, black, pygame.Rect((col + 5) * squareLength, (self.nextPiece.height() - row) * squareLength, squareLength, squareLength))
        pygame.display.update()

        pygame.event.pump()  # to tell the operating system that pygame is still running

game = Game()
while game.isAlive():
    # make random moves for now
    game.currentPiece.ori = random.randint(0, len(game.currentPiece.piece.bitMaps) - 1)
    game.currentPiece.col = random.randint(0, gridWidth - game.currentPiece.width())

    game.dropPiece()

    game.render()
    print("done rendering")

# eventually I should should make sure that when a human user rotates a piec when pushed against the right side the piece never go off the grid
