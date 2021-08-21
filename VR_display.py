# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 07:53:07 2020

@author: Daren
"""

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
import cv2
import numpy as np
import dlib


cap = cv2.VideoCapture(1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

face = None
p1 = (0,0)
p2 = (0,0)
nose_hist = np.zeros((10,2))


verticies = (
    (100, -100, -100),
    (100, 100, -100),
    (-100, 100, -100),
    (-100, -100, -100),
    (100, -100, 100),
    (100, 100, 100),
    (-100, -100, 100),
    (-100, 100, 100)
    )


edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
    )


    
colors = (
    (1,0,0),
    (0,1,0),
    (0,0,1),
    (1,1,0),
    (1,0,1),
    (0,1,1),
    (0.5,1,0.5),
    )

surfaces = (
    (0,1,2,3),
    (3,2,7,6),
    (6,7,5,4),
    (4,5,1,0),
    (1,5,7,2),
    (4,0,3,6)
    )


def Cube():
    glBegin(GL_QUADS)
    x = 0
        
    #Right
    glColor3fv((255,255,255))
    glVertex3fv(verticies[4])
    glVertex3fv(verticies[5])
    glColor3fv((0,0,0))
    glVertex3fv(verticies[1])
    glVertex3fv(verticies[0])
    
    #Left
    glColor3fv((255,255,255))
    glVertex3fv(verticies[6])
    glVertex3fv(verticies[7])
    glColor3fv((0,0,0))
    glVertex3fv(verticies[2])
    glVertex3fv(verticies[3])
    
    #Top
    glColor3fv((255,255,255))
    glVertex3fv(verticies[7])
    glVertex3fv(verticies[5])
    glColor3fv((0,0,0))
    glVertex3fv(verticies[1])
    glVertex3fv(verticies[2])
    
    #Bottom
    glColor3fv((255,255,255))
    glVertex3fv(verticies[4])
    glVertex3fv(verticies[6])
    glColor3fv((0,0,0))
    glVertex3fv(verticies[3])
    glVertex3fv(verticies[0])
    glEnd()

    glBegin(GL_LINES)
    glColor3fv((0,0,0))
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()

def Circum(depth):
    verticies1 = (
        (100, -100, -depth),
        (100, 100, -depth),
        (-100, 100, -depth),
        (-100, -100, -depth))
    
    edges1 = (
        (0,1),
        (0,3),
        (2,1),
        (2,3)
        )
    glColor3fv((255,255,255))
    glBegin(GL_LINES)
    for edge in edges1:
        for vertex in edge:
            glVertex3fv(verticies1[vertex])
    glEnd()

def ULWall(width):
    
    verticies1 = ((width, -100, -100),
    (width, 100, -100),
    (width, -100, 100),
    (width, 100, 100))
    
    edges1 = ((0,2),
    (0,1),
    (3,1))
    
    glColor3fv((255,255,255))
    glBegin(GL_LINES)
    
    
    for edge in edges1:
        for vertex in edge:
            glVertex3fv(verticies1[vertex])
    glEnd()

def LRWall(width):
    
    verticies1 = ((100, width, -100),
    (-100, width, -100),
    (100, width, 100),
    (-100, width, 100),)
    
    edges1 = ((0,1),
    (0,2),
    (1,3))
    glColor3fv((255,255,255))
    glBegin(GL_LINES)
    for edge in edges1:
        for vertex in edge:
            glVertex3fv(verticies1[vertex])
    glEnd()

def DrawCircle(radius, side_num, edge_only,x ,y, z, color):

    if(edge_only):
        glBegin(GL_LINE_LOOP)
    else:
        glBegin(GL_POLYGON)
        
    glColor3fv(colors[color])
    for vertex in range(0, side_num):
        angle  = float(vertex) * 2.0 * np.pi / side_num
        glVertex3f(np.cos(angle)*radius+x, np.sin(angle)*radius+y, z)
    
    glEnd();
    
def DrawLine(x ,y, z):
    glBegin(GL_LINES)
    glColor3fv((255,255,255))
    glVertex3fv((x,y,z))
    glColor3fv((0,0,0))
    glVertex3fv((x,y,-100))
    glEnd()
    
def main():
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    
    nose_hist = np.zeros((15,2))
    a = 100
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        faces = detector(gray)
        
     
    
        biggest_face_area = 0
        for face_id in faces:
            x1 = face_id.left()
            y1 = face_id.top()
            x2 = face_id.right()
            y2 = face_id.bottom()
            face_area = abs((x2-x1)*(y2-y1))
            if face_area > biggest_face_area:
                biggest_face_area = face_area
                face = face_id
                
        if len(faces) > 0:
            landmarks = predictor(gray, face)
        
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                if n in [36,45,33,8,48,54]:
                    cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
        
        
            #2D image points. If you change the image, you need to change vector
            image_points = np.array([
                                        (landmarks.part(30).x, landmarks.part(30).y),     # Nose tip
                                        (landmarks.part(8).x, landmarks.part(8).y),       # Chin
                                        (landmarks.part(45).x, landmarks.part(45).y),     # Left eye left corner
                                        (landmarks.part(36).x, landmarks.part(36).y),     # Right eye right corne
                                        (landmarks.part(54).x, landmarks.part(54).y),     # Left Mouth corner
                                        (landmarks.part(48).x, landmarks.part(48).y)      # Right mouth corner
                                    ], dtype="double")        
        
            nose = np.array([image_points[0]])

            nose_hist = np.append(nose_hist,nose,0)
            nose_hist = np.delete(nose_hist,0,0)
            nose = tuple((nose_hist.sum(0)/nose_hist.shape[0]).astype(int))
            
            pa = np.array([-100, -100, 0])
            pb = np.array([100,  -100, 0])
            pc = np.array([-100,  100, 0])
            
            pe = np.array([-(nose[0]-320)*100/320, -(nose[1]-240)*100/240, a])
            
            n = 0.1
            f = 500
 
           
            vr = pb - pa
            vu = pc - pa
            vn = np.cross(vr, vu)
            

            vr = vr/np.linalg.norm(vr)
            vu = vu/np.linalg.norm(vu)
            vn = vn/np.linalg.norm(vn)
            
            va = pa - pe
            vb = pb - pe
            vc = pc - pe            
            

            
            d = -np.dot(vn,va)
            l =  np.dot(vr,va)*n/d
            b =  np.dot(vu,va)*n/d
            r =  np.dot(vr,vb)*n/d
            t =  np.dot(vu,vc)*n/d
            
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
        
            #transform projection to that of our eye
            glFrustum(l, r, b, t, n, f)
            
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()   
            glTranslatef(-pe[0],-pe[1],-pe[2])      
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                cap.release()
                cv2.destroyAllWindows()
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    a += 10
                    print(pe[2])
                if event.key == pygame.K_UP:
                    a -= 10
                    print(pe[2])
                    
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Cube()
        for depth in range(0,100,20):
            Circum(depth)
        for width in range(-100,100,20):
            ULWall(width)
            LRWall(width)
        
        DrawLine(10, 0, 5)   
        DrawLine( 0, -45, 10)
        DrawLine(20, -30, 20)
        DrawLine(-30, 10, 30)
        DrawLine(-40, -30, 40) 
        DrawLine(10,10,60)
        DrawLine(30, 40, 70)
        
        
        DrawCircle(10, 50, False, 10, 0, 5, 6)
        DrawCircle(10, 50, False, 0, -45, 10, 4)
        DrawCircle(10, 50, False, 20, -30, 20, 2)
        DrawCircle(10, 50, False, -30, 10, 30, 1)
        DrawCircle(10, 50, False, -40, -30, 40, 3)
        DrawCircle(10, 50, False, 10, 10, 60, 0)
        DrawCircle(10, 50, False, 30, 40, 70, 5)
        
        pygame.display.flip()
        pygame.time.wait(10)


main()