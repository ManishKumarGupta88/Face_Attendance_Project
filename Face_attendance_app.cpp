//|---------------------------------------------------------------------------------------------------------------------------------|
//|                                          Welcome To My Face Atandance(Detection) App                                            |
//|---------------------------------------------------------------------------------------------------------------------------------|
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>


using namespace cv;
using namespace std;
namespace fs = std::filesystem;
//|-------------------------------|
//|Class to handle student data   |
//|-------------------------------|
class Student {
public:
    string name;
    Mat face;
};
//|--------------------------------------|
//|Class for Face Attendance Application |
//|--------------------------------------|
class FaceAttendanceApp {
private:
    CascadeClassifier face_cascade;
    VideoCapture cap;
    vector<Student> students;
    const string referenceFolderPath = "C:\\Users\\manis\\Downloads\\reference_faces\\";
    const string attendanceFolderPath = "C:\\Users\\manis\\Downloads\\attandance\\";

public:

    // Constructor to initialize camera and Haar cascade
    FaceAttendanceApp() {
        string cascadePath = "C:\\Users\\manis\\Downloads\\haarcascade_frontalface_default.xml";
        if (!face_cascade.load(cascadePath)) {
            cerr << "Error loading Haarcascade file\n";
            exit(1);
        }

        if (!cap.open(0)) {
            cerr << "Error opening webcam\n";
            exit(1);
        }

        // Create folders if they don't exist
        if (!fs::exists(referenceFolderPath)) fs::create_directory(referenceFolderPath);
        if (!fs::exists(attendanceFolderPath)) fs::create_directory(attendanceFolderPath);

        loadReferenceFaces();
    }

    // Detect face in the given frame
    Mat detectFace(Mat& img) {
        vector<Rect> faces;
        Mat gray, faceROI;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        face_cascade.detectMultiScale(gray, faces, 1.1, 4);

        if (!faces.empty()) {
            faceROI = gray(faces[0]);
        }
        return faceROI;
    }

    // Register new student faces
    void registerStudents() {
        int n;
        cout << "Enter number of students to register: ";
        cin >> n;

        for (int i = 0; i < n; i++) {
            string name;
            cout << "Enter name of student " << (i + 1) << ": ";
            cin >> name;

            Mat frame, faceROI;
            while (true) {
                cap >> frame;
                faceROI = detectFace(frame);

                if (!faceROI.empty()) {
                    resize(faceROI, faceROI, Size(100, 100));
                    imwrite(referenceFolderPath + name + ".jpg", faceROI);
                    students.push_back({ name, faceROI });
                    cout << "Captured and saved face for " << name << ".\n";
                    break;
                }
                imshow("Capture Student Face", frame);
                if (waitKey(30) >= 0) break;
            }
        }
        destroyAllWindows();
    }

    // Load reference faces from folder
    void loadReferenceFaces() {
        for (const auto& entry : fs::directory_iterator(referenceFolderPath)) {
            string filePath = entry.path().string();
            Mat face = imread(filePath, IMREAD_GRAYSCALE);
            if (!face.empty()) {
                string name = entry.path().stem().string();
                resize(face, face, Size(100, 100));
                students.push_back({ name, face });
            }
        }
    }

    // Mark attendance
    void markAttendance() {
        if (students.empty()) {
            cout << "No students registered. Please register students first.\n";
            return;
        }

        string attendanceFilePath = attendanceFolderPath + "attendance.txt";
        ofstream attendanceFile(attendanceFilePath);
        if (!attendanceFile.is_open()) {
            cerr << "Error opening attendance file\n";
            return;
        }

        vector<string> presentStudents;
        while (true) {
            Mat frame, faceROI;
            cap >> frame;
            faceROI = detectFace(frame);

            if (!faceROI.empty()) {
                resize(faceROI, faceROI, Size(100, 100));
                for (const auto& student : students) {
                    if (norm(student.face, faceROI, NORM_L2) < 5000) {
                        if (find(presentStudents.begin(), presentStudents.end(), student.name) == presentStudents.end()) {
                            presentStudents.push_back(student.name);
                            attendanceFile << student.name << " -> Present\n";
                            cout << student.name << " is present.\n";
                        }
                    }
                }
            }

            imshow("Mark Attendance", frame);
            if (waitKey(30) >= 0) break;
        }
        attendanceFile.close();
        destroyAllWindows();
    }

    // Update a student's face
    void updateStudentFace() {
        if (students.empty()) {
            cout << "No students registered. Please register students first.\n";
            return;
        }

        string name;
        cout << "Enter the name of the student to update face: ";
        cin >> name;

        for (auto& student : students) {
            if (student.name == name) {
                Mat frame, faceROI;
                while (true) {
                    cap >> frame;
                    faceROI = detectFace(frame);

                    if (!faceROI.empty()) {
                        resize(faceROI, faceROI, Size(100, 100));
                        imwrite(referenceFolderPath + name + ".jpg", faceROI);
                        student.face = faceROI;
                        cout << "Updated face for " << name << ".\n";
                        break;
                    }
                    imshow("Update Student Face", frame);
                    if (waitKey(30) >= 0) break;
                }
                destroyAllWindows();
                return;
            }
        }
        cout << "Student not found.\n";
    }

    // Delete a student's face
    void deleteStudentFace() {
        if (students.empty()) {
            cout << "No students registered. Please register students first.\n";
            return;
        }

        string name;
        cout << "Enter the name of the student to delete face: ";
        cin >> name;

        auto it = remove_if(students.begin(), students.end(),
            [&name](const Student& s) { return s.name == name; });
        if (it != students.end()) {
            students.erase(it, students.end());
            fs::remove(referenceFolderPath + name + ".jpg");
            cout << "Deleted face for " << name << ".\n";
        }
        else {
            cout << "Student not found.\n";
        }
    }

    // View registered students
    void viewStudents() const {
        if (students.empty()) {
            cout << "No students registered yet.\n";
            return;
        }
        cout << "Registered Students:\n";
        for (const auto& student : students) {
            cout << student.name << "\n";
        }
    }

    // Main menu
    void menu() {
        while (true) {
            int choice;
            cout << "\n--- Face Attendance App ---\n";
            cout << "1. Register Students\n";
            cout << "2. Mark Attendance\n";
            cout << "3. Update Student Face\n";
            cout << "4. Delete Student Face\n";
            cout << "5. View Registered Students\n";
            cout << "6. Exit\n";
            cout << "Enter your choice: ";
            cin >> choice;

            switch (choice) {
            case 1:
                registerStudents();
                break;
            case 2:
                markAttendance();
                break;
            case 3:
                updateStudentFace();
                break;
            case 4:
                deleteStudentFace();
                break;
            case 5:
                viewStudents();
                break;
            case 6:
                cout << "Exiting...\n";
                return;
            default:
                cout << "Invalid choice. Try again.\n";
            }
        }
    }
};

int main() {
    FaceAttendanceApp app;
    app.menu();
    return 0;
}
