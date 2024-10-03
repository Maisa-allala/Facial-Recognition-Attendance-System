from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk


class Student:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")

     # first image
        img1 = Image.open(r"C:\Users\maiss\OneDrive\Desktop\Facial Recognition Attendance System\images\2.jpeg")
        img1= img1.resize((500,130), Image.ANTIALIAS)
        self.photoimg1=ImageTk.PhotoImage(img1)
        f_lbl = Label(self.root, image=self.photoimg1)
        f_lbl.place(x=0, y=0, width=500, height=130)

        # second image
        img2 = Image.open(r"C:\Users\maiss\OneDrive\Desktop\Facial Recognition Attendance System\images\1.jpg")
        img2= img2.resize((500,130), Image.ANTIALIAS)
        self.photoimg2=ImageTk.PhotoImage(img2)
        f_lbl = Label(self.root, image=self.photoimg2)
        f_lbl.place(x=500, y=0, width=500, height=130)

         # Third image
        img3 = Image.open(r"C:\Users\maiss\OneDrive\Desktop\Facial Recognition Attendance System\images\3.jpg")
        img3= img3.resize((560,130), Image.ANTIALIAS)
        self.photoimg3=ImageTk.PhotoImage(img3)
        f_lbl = Label(self.root, image=self.photoimg3)
        f_lbl.place(x=1000, y=0, width=560, height=130)   

        
         # bg image
        img4 = Image.open(r"C:\Users\maiss\OneDrive\Desktop\Facial Recognition Attendance System\images\3.jpg")
        img4= img4.resize((1530,710), Image.ANTIALIAS)
        self.photoimg4=ImageTk.PhotoImage(img4)

        bg_img = Label(self.root, image=self.photoimg4)
        bg_img.place(x=0, y=130, width=1530, height=710)

        title_lbl = Label(bg_img, text = "STUDENT MANAGEMENT SYSTEM", font=("Times New Roman", 35, "bold"), bg="white", fg="darkgreen")
        title_lbl.place(x=0, y=0,width=1530, height=45)


        main_frame=Frame(bg_img, bd=2)
        main_frame.place(x=20, y=50, width=1480, height=600)
        # left label frame
        Left_Frame = LabelFrame(main_frame, bd=2, bg="white" ,relief=RIDGE, text="Student Details", font=("times new roman", 12, "bold"))
        Left_Frame.place(x=10, y=10, width=730, height=580 )


        img_left = Image.open(r"C:\Users\maiss\OneDrive\Desktop\Facial Recognition Attendance System\images\3.jpg")
        img_left= img_left.resize((720,130), Image.ANTIALIAS)
        self.photoimg_left=ImageTk.PhotoImage(img_left)
        f_lbl = Label(Left_Frame, image=self.photoimg_left)
        f_lbl.place(x=5, y=0, width=720, height=130)   
        
        # current course
        current_course_frame = LabelFrame(Left_Frame, bd=2, bg="white" ,relief=RIDGE, text="Current course information", font=("times new roman", 13, "bold"))
        current_course_frame.place(x=5, y=135, width=720, height=150 )

        # Department
        dep_label = Label(current_course_frame, text="Department", font=("times new roman", 13, "bold"), bg="white")
        dep_label.grid(row=0, column=0, padx=10,  sticky=W )
        dep_combo = ttk.Combobox(current_course_frame, font=("times new roman", 13, "bold"),  state="readonly", width=20)
        dep_combo["values"]=("Select Department", "Computer", "IT", "Civil", "Mechnical")
        dep_combo.current(0)
        dep_combo.grid(row=0, column=1, padx=2, pady=10, sticky=W)

        # Course
        course_label = Label(current_course_frame, text="Course", font=("times new roman", 13, "bold"), bg="white")
        course_label.grid(row=0, column=2, padx=10,  sticky=W )
        course_combo = ttk.Combobox(current_course_frame, font=("times new roman", 13, "bold"),  state="readonly", width=20)
        course_combo["values"]=("Select Course", "FE", "SE", "TE", "BE")
        course_combo.current(0)
        course_combo.grid(row=0, column=3, padx=2, pady=10, sticky=W)

         # Year
        course_label = Label(current_course_frame, text="Year", font=("times new roman", 13, "bold"), bg="white")
        course_label.grid(row=1, column=0, padx=10,  sticky=W )
        course_combo = ttk.Combobox(current_course_frame, font=("times new roman", 13, "bold"),  state="readonly", width=20)
        course_combo["values"]=("Select Year", "2020-21", "2021-22", "2022-23", "2023-24")
        course_combo.current(0)
        course_combo.grid(row=1, column=1, padx=2, pady=10, sticky=W)


         # Semester
        course_label = Label(current_course_frame, text="Semester", font=("times new roman", 13, "bold"), bg="white")
        course_label.grid(row=1, column=2, padx=10,  sticky=W )
        course_combo = ttk.Combobox(current_course_frame, font=("times new roman", 13, "bold"),  state="readonly", width=20)
        course_combo["values"]=("Select Semester", "Semester-1", "Semester-2")
        course_combo.current(0)
        course_combo.grid(row=1, column=3, padx=2, pady=10, sticky=W)


        # Right label frame
        Right_Frame = LabelFrame(main_frame, bd=2, bg="white" ,relief=RIDGE, text="Student Details", font=("times new roman", 12, "bold"))
        Right_Frame.place(x=750, y=10, width=660, height=580 )

if __name__ == "__main__":
    root = Tk()
    obj = Student(root)
    root.mainloop()
