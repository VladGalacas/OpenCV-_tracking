import cv2
import sys
import time
import matplotlib.pyplot as plt


def fps(type_tracker: int, way_video: str, box_object: tuple):
    tracker_type = tracker_types[type_tracker]

    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()  # neural net
    elif tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == "CSRT":
        tracker = cv2.legacy.TrackerCSRT_create()

    # Read video

    video = cv2.VideoCapture(way_video)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()

    # записать в файл
    frame_height, frame_width = frame.shape[:2]
    output = cv2.VideoWriter(f'{tracker_type}_hidden.avi',
                             cv2.VideoWriter_fourcc(*'XVID'), 60.0,
                             (frame_width, frame_height), True)

    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Define an initial bounding box
    bbox = box_object  # ЛЕВЫЙ ВЕРХНИЙ УГОЛ (X,Y), ШИРИНА, ДЛИНА, можно программой PAINT, кадр можно вырезать save_frame.py

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    all_fps = []
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            print("Frame is bad")
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        all_fps.append(fps)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display average FPS to N frames
        cv2.putText(frame, f'Average FPS up to {len(all_fps)} frames: {round(sum(all_fps) / len(all_fps))}', (100, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", frame)

        output.write(frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff  # k = cv2.waitKey(0) - тогда будет показ по каждому кадру
        if k == 27:  # 27 - в таблице ASCII - Esc
            break

    return round(sum(all_fps) / len(all_fps)), tracker_type

    video.release()
    output.release()
    cv2.destroyAllWindows()


def plot_avg_fps(way_image: str, way_video: str, box_object: tuple) -> None:
    all_avg_fps = []
    for i in range(len(tracker_types)):
        avg_fps, tracker = fps(i, way_video, box_object)
        all_avg_fps.append(avg_fps)
        print(f'For {tracker} average FPS: {avg_fps}')

    sorted_fps = sorted(list(zip(tracker_types, all_avg_fps)), key=lambda x: -x[1])
    plt.bar([dig[0] for dig in sorted_fps], [trac[1] for trac in sorted_fps])
    plt.ylabel('Average FPS')
    plt.xlabel('Trackers')
    plt.title('Comparison of FPS for different methods')
    plt.savefig(f'{way_image}\\Comparison of FPS for different methods.png')


if __name__ == '__main__':
    tracker_types = ['KCF', 'MOSSE', 'TLD', 'MIL', 'MEDIANFLOW', 'GOTURN', 'BOOSTING', 'CSRT']  # MEDIANFLOW, MOSSE
    plot_avg_fps(box_object=(1394, 447, 51, 69),
                 way_image='',
                 way_video='')

