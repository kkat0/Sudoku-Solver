import sys
import os
import numpy as np
import cv2
from keras.models import load_model

def main():
    if len(sys.argv) < 2:
        print("Usage: 'python main.py [IMAGE_FILE_PATH]'")
        exit(1)

    img_path = sys.argv[1]
    img = cv2.imread(img_path)

    found, extracted, img = extract(img)
    cv2.imshow('original', img)

    if found:
        cv2.imshow("extracted", extracted)
        board = recognize_board(extracted)
        solved, filled = solve_sudoku(board)
        show_board('board', board)
        show_board('filled', filled)        

    else:
        not_found_img = np.zeros((252, 252))
        cv2.putText(back, 'Not found', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, 255, 1, cv2.LINE_AA)
        cv2.imshow("Not found", not_found_img)    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_board(title, board):
    board_img = np.ones((500, 500, 3))

    if board is None:
        board_img = np.zeros((500, 500, 3))
        cv2.putText(board_img, 'Can not solve.', (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 5, cv2.LINE_AA)
    else:
        ls = np.linspace(0, 500, 10).astype(np.int)

        for i in range(10):
            board_img = cv2.line(board_img, (ls[i], 0), (ls[i], 500), (0, 0, 0), 3)
            board_img = cv2.line(board_img, (0, ls[i]), (500, ls[i]), (0, 0, 0), 3)

        for i in range(9):
            for j in range(9):
                tx = ls[j] + 8
                ty = ls[i] + 50
                if board[i, j] != 0:
                    cv2.putText(board_img, str(board[i, j]), (tx, ty), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 5, cv2.LINE_AA)
        
    cv2.imshow(title, board_img)

def extract(img):
    height = 480
    width = int(height * img.shape[1] / img.shape[0])
    img = cv2.resize(img, (width, height))
    
    a = cv2.bilateralFilter(img, 11, 21, 21)
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    a = cv2.adaptiveThreshold(a, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 15)
    
    _, contours, _ = cv2.findContours(a, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects = list(filter(lambda x: len(x) == 4, [cv2.approxPolyDP(cnt, cv2.arcLength(cnt, True)*0.05, True) for cnt in contours]))
    
    if len(rects) == 0:
        return False, None, img
    
    largest_rect = sorted(rects, key=lambda x: cv2.contourArea(x), reverse=True)[0].reshape(4, 2)

    rect_area = cv2.contourArea(largest_rect)

    if rect_area < (min(img.shape[:2]) / 3) ** 2:
        return False, None, img
    
    idx = 0
    min_dist = float('inf')
    
    for i, p in enumerate(largest_rect):
        dist = p[0] + p[1]
        
        if dist < min_dist:
            min_dist = dist
            idx = i
    
    p_src = np.ndarray(largest_rect.shape, dtype=np.float32)
    p_dst = np.array([
        [0, 0],
        [0, 252],
        [252, 252],
        [252, 0],
    ], dtype=np.float32)
    
    for i in range(4):
        p_src[i] = largest_rect[(idx + i) % 4]
    
    tf_mat = cv2.getPerspectiveTransform(p_src, p_dst)
    dst_img = cv2.warpPerspective(a, tf_mat, (252, 252))

    return True, dst_img, img

def recognize_board(img):
    digit_clf = load_model('cell_recognizer.h5')
    cells = np.ndarray((81, 28, 28, 1))
    
    for i in range(9):
        for j in range(9):
            cells[9 * i + j] = img[28*i:28*(i+1), 28*j:28*(j+1)].reshape(28, 28, 1)

    pre = digit_clf.predict(cells.astype(np.float32) / 255, verbose=1)
    
    return pre.argmax(axis=1).reshape(9, 9)

def solve_sudoku(board):
    return solve_dfs(board, 0)

def solve_dfs(board, i):    
    if i == 81:
        return True, board.copy()
    
    if board[i // 9, i % 9] != 0:
        return solve_dfs(board, i + 1)
    
    bx = (i % 9) // 3
    by = i // 27
    
    for n in range(1, 10):
        ok = True
        for j in range(9):
            if board[i // 9, j] == n or board[j, i % 9] == n or board[3 * by + (j // 3), 3 * bx + (j % 3)] == n:
                ok = False
        
        if not ok:
            continue
        
        board[i // 9, i % 9] = n   
        ret = solve_dfs(board, i + 1) 
        board[i // 9, i % 9] = 0

        if ret[0] == True:
            return ret
    
    return False, None

if __name__ == '__main__':
    main()