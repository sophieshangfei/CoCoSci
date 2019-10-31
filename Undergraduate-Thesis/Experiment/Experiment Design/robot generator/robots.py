from turtle import *
import turtle
title("robot")

r = Turtle()
image = "by.gif"
r.screen.bgpic(image)
r.screen.setup(1000, 1000)

# draw a circle head
def draw_head(radius):
	turtle=Turtle()
	turtle.up()
	turtle.goto(0,100)
	turtle.down()
	turtle.color("black")
	turtle.width(width=5)
	turtle.end_fill()
	turtle.circle(radius)
	turtle.hideturtle()
draw_head(50)
# 90, 70, 50

# calculate eye position
def eye_coordinate(radius):
	y_coordinate = 100 + 1.3*radius
	x_coordinate_l = -(1/3)*radius
	x_coordinate_r = (1/3)*radius
	return [(x_coordinate_l, y_coordinate), (x_coordinate_r, y_coordinate)]

# draw eyes
def draw_eyes(radius):
	eye_l = Turtle()
	c = eye_coordinate(radius)
	eye_l.up()
	eye_l.goto(c[0])
	eye_l.down()
	eye_l.color("black")
	eye_l.width(width = 5)
	eye_l.end_fill()
	eye_l.circle(5)
	eye_r = Turtle()
	eye_r.up()
	eye_r.goto(c[1])
	eye_r.down()
	eye_r.color("black")
	eye_r.width(width = 5)
	eye_r.end_fill()
	eye_r.circle(5)
	eye_r.hideturtle()
	eye_l.hideturtle()
draw_eyes(50)


# draw a sqaure body
def draw_square():
    """ draw square for turtles """

    # to draw a square you want to : move forward, turn right,
    #  move forward, turn right,move forward turn right
    square = Turtle()
    square.up()
    square.goto(-100, 100)
    square.down()
    square.color("blue")
    square.begin_fill()
    square.width(width=5)
    square.forward(200)  # forward takes a number which is the distance to move
    square.right(90)  # turn right
    square.forward(200)
    square.right(90)
    square.forward(200)
    square.right(90)
    square.forward(200)
    square.right(90)
    square.end_fill()
    square.hideturtle()
    
draw_square()

# draw two feet
def draw_feet():
	feet = Turtle()
	feet.up()
	feet.goto(-50, -150)
	feet.down()
	feet.width(width=5)
	feet.color("#e06696")
	feet.begin_fill()
	feet.circle(25)
	feet.hideturtle()
	feet.up()
	feet.goto(50, -150)
	feet.down()
	feet.width(width=5)
	feet.circle(25)
	feet.end_fill()
	feet.hideturtle()
draw_feet()


#draw control panel
def draw_panel(size):
	panel = Turtle()
	panel.up()
	panel.goto(-size,20)
	panel.down()
	panel.color("black")
	panel.begin_fill()
	panel.width(width=5)
	panel.forward(2*size)
	panel.right(90)
	panel.forward(size)
	panel.right(90)
	panel.forward(2*size)
	panel.right(90)
	panel.forward(size)
	panel.right(90)
	panel.end_fill()
	panel.hideturtle()
draw_panel(80)
# 40&80; 60&120; 80&160
# draw two arms
def draw_arm(angle):
	angle_1 = angle + 180
	angle_2 = 360 - angle
	L_arm = Turtle()
	L_arm.up()
	L_arm.setheading(350)
	L_arm.goto(-100, 100)
	L_arm.down()
	L_arm.color("black")
	L_arm.width(width=5)
	L_arm.forward(10)
	L_arm.right(90)
	L_arm.forward(100)
	L_arm.right(90)
	L_arm.forward(10)
	L_arm.right(90)
	L_arm.forward(100)
	L_arm.right(90)
	L_arm.hideturtle()

	R_arm = Turtle()
	R_arm.up()
	R_arm.setheading(550)
	R_arm.goto(100, 100)
	R_arm.down()
	R_arm.color("black")
	R_arm.width(width=5)
	R_arm.forward(10)
	R_arm.left(90)
	R_arm.forward(100)
	R_arm.left(90)
	R_arm.forward(10)
	R_arm.left(90)
	R_arm.forward(100)
	R_arm.left(90)
	R_arm.hideturtle()

	
draw_arm(90)


r.getscreen()._root.mainloop()









