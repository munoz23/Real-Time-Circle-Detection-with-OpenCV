import cv2
import numpy as np
import math

class DetectionCircle:
    def __init__(self):
        self.t = 100
        self.w = int(640)
        self.sc = 1
        self.md = 30
        self.at = 40
        self.capture = cv2.VideoCapture(0)
        self.measure_option = "None"  # Opción por defecto: Ninguna medida

    def closeCamera(self):
        cv2.destroyAllWindows()

    def calculate_distance(self, x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def calculate_magnitude(self, x, y):
        return np.sqrt(x ** 2 + y ** 2)

    def calculate_angle(self, x1, y1, x2, y2):
        return math.degrees(math.atan2(y2 - y1, x2 - x1))

    def main(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                continue

            img_height, img_width, depth = frame.shape
            scale = self.w / img_width
            h = int(img_height * scale)
            frame = cv2.resize(frame, (self.w, h))

            colorCv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bluredCv2 = cv2.medianBlur(colorCv2, 15)

            grid = np.zeros([2*h, 2*self.w, 3], np.uint8)
            grid[0:h, 0:self.w] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            grid[h:2*h, 0:self.w] = np.dstack([cv2.Canny(colorCv2, self.t / 2, self.t)] * 3)
            grid[0:h, self.w:2*self.w] = np.dstack([bluredCv2] * 3)
            grid[h:2*h, self.w:2*self.w] = np.dstack([cv2.Canny(bluredCv2, self.t / 2, self.t)] * 3)

            cv2.imshow('Detección en tiempo real', grid)

            circles = cv2.HoughCircles(bluredCv2, cv2.HOUGH_GRADIENT, self.sc, self.md, self.t, self.at, minRadius=0, maxRadius=0)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                if len(circles[0]) >= 2:
                    # Dibujar círculos y unirlos con líneas
                    for i in range(len(circles[0])):
                        x1, y1, radius1 = circles[0][i]
                        cv2.circle(frame, (x1, y1), radius1, (0, 0, 255), 4)
                        cv2.circle(frame, (x1, y1), 1, (0, 0, 255), 4)
                        center1 = (x1, y1)
                        for j in range(i + 1, len(circles[0])):
                            x2, y2, radius2 = circles[0][j]
                            cv2.circle(frame, (x2, y2), radius2, (0, 0, 255), 4)
                            cv2.circle(frame, (x2, y2), 1, (0, 0, 255), 4)
                            center2 = (x2, y2)
                            # Dibujar una línea entre los centros de los círculos
                            cv2.line(frame, center1, center2, (0, 255, 0), 2)

                            if self.measure_option == "Distance":
                                # Calcular la distancia entre los centros de los círculos
                                distance = self.calculate_distance(x1, y1, x2, y2)
                                # Mostrar la distancia calculada sobre la línea
                                text_position = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                                cv2.putText(frame, f"Distance: {distance:.2f}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                            if self.measure_option == "Magnitude":
                                # Calcular la magnitud del vector formado por los centros de los círculos
                                dx, dy = x2 - x1, y2 - y1
                                magnitude = self.calculate_magnitude(dx, dy)
                                # Mostrar la magnitud calculada sobre la línea
                                text_position = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                                cv2.putText(frame, f"Magnitude: {magnitude:.2f}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                            if self.measure_option == "Angle":
                                # Calcular el ángulo entre los centros de los círculos
                                angle = self.calculate_angle(x1, y1, x2, y2)
                                # Mostrar el ángulo calculado sobre la línea
                                text_position = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                                cv2.putText(frame, f"Angle: {angle:.2f} degrees", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow('Image with detected circles', frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key == ord('d'):
                self.measure_option = "Distance"
            elif key == ord('m'):
                self.measure_option = "Magnitude"
            elif key == ord('a'):
                self.measure_option = "Angle"
            elif key == ord('n'):
                self.measure_option = "None"

        self.capture.release()
        self.closeCamera()

if __name__ == "__main__":
    DetectionCircle().main()