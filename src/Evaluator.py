import cv2
import numpy as np
import os

from PathsProvider import PathsProvider
from HogClassifier import HogClassifier

class Evaluator:

    def __init__(self):
        self.paths = PathsProvider()
        self.classifier = HogClassifier()

    def evaluate(self, images_folder_path, labels_folder_path, iou_threshold=0.5):
        images_names = os.listdir(images_folder_path)
        gt_count_list = []
        detect_count_list = []
        tp_list = []
        fp_list = []
        fn_list = []
        iou_mean_list = []

        for image_name in images_names:
            image_path = os.path.join(images_folder_path, image_name)
            label_name = image_name.split('.')[0]

            gt_bboxes = self._read_bboxes_file(f"{labels_folder_path}/{label_name}.txt")
            _, detect_bboxes = self.classifier.find_glyphs(image_path)
            tp, fp, fn, iou_mean = self.evaluate_bboxes(gt_bboxes, detect_bboxes, image_path, iou_threshold, True)

            tp_list += [tp]
            fp_list += [fp]
            fn_list += [fn]
            iou_mean_list += [iou_mean]
            gt_count_list.append(len(gt_bboxes))
            detect_count_list.append(len(detect_bboxes))

        print(f"* Hay {np.sum(gt_count_list)} glifos en total (groundtruth). Media: {np.mean(gt_count_list)} por pagina")
        print(f"* Find contours ha detectado {np.sum(detect_count_list)} glifos.  Media: {np.mean(detect_count_list)} por pagina")
        print(f"    * {np.sum(tp_list)}/{np.sum(detect_count_list)} ({np.sum(tp_list)*100/np.sum(detect_count_list): .2f}% ) coinciden con glifos en un IoU > {iou_threshold}")
        print(f"        * {np.sum(tp_list)}/{np.sum(gt_count_list)} ({np.sum(tp_list)*100/np.sum(gt_count_list): .2f}% ) detecciones coinciden con groundtruth ")
        print(f"    * {np.sum(fp_list)}/{np.sum(detect_count_list)}  ({np.sum(fp_list)*100/np.sum(detect_count_list): .2f}% ) NO coinciden con glifos en un IoU > {iou_threshold}")
        print(f"    * Media de IoUs: {np.mean(iou_mean_list): .2f}")



    def evaluate_bboxes(self, groundtruth_bboxes, detected_bboxes, image_path, iou_threshold, verbose = False):
        """
        Calculation for IoU's of bounding boxes for a specific image.

        :param groundtruth_bboxes: bounding boxes manually annotated.
        :param detected_bboxes: bounding boxes detected by the algorithm.
        :param image_path: path of evaluated image.
        :param iou_threshold: Threshold for IoU computing.
        :param verbose: verbose will print information
        :return: true positives, false positives, false negatives.
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        n_detected = len(detected_bboxes)
        ious_list = []
        image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)

        for gt_bbox in groundtruth_bboxes:
            gx1, gy1, gx2, gy2 = gt_bbox
            cv2.rectangle(image, (gx1, gy1), (gx2, gy2), (0, 0, 255), 1)
            overlapped_bboxes = []
            for dt_bbox in detected_bboxes:
                iou = self._intersection_over_union(dt_bbox, gt_bbox)
                if iou > iou_threshold:
                    overlapped_bboxes.append(dt_bbox)
                    cv2.rectangle(image, (dt_bbox[0], dt_bbox[1]), (dt_bbox[2], dt_bbox[3]), (0, 255, 0), 1)
                    true_positives += 1
                    ious_list.append(iou)

            detected_bboxes = [bbox for bbox in detected_bboxes if bbox not in overlapped_bboxes]
            if len(overlapped_bboxes) == 0:
                cv2.rectangle(image, (gx1, gy1), (gx2, gy2), (255, 255, 0), 1)
                false_negatives += 1

        for dt_bbox in detected_bboxes:
            cv2.rectangle(image, (dt_bbox[0], dt_bbox[1]), (dt_bbox[2], dt_bbox[3]), (255, 0, 0), 1)
            false_positives += 1

        cv2.imwrite(f"{self.paths.RESULTS}/contours/images/{os.path.basename(image_path)}", image)

        if verbose:
            print(f"En la imagen {os.path.basename(image_path)} hay {len(groundtruth_bboxes)} detecciones (groundtruth):")
            print(f"   * Hubo {false_negatives} glifos que no se detectaron.")
            print(f"   * El algoritmo hizo {n_detected} detecciones:")
            print(
                f"      * {true_positives}/{n_detected} ({true_positives * 100 / n_detected: .2f}%) coinciden con glifos (IoU > {iou_threshold}) -> {true_positives}/{len(groundtruth_bboxes)} ({true_positives * 100 / len(groundtruth_bboxes): .2f}%) del groundtruth")
            print(
                f"      * {false_positives}/{n_detected} ({false_positives * 100 / n_detected: .2f}%) NO coinciden con glifos (IoU <= {iou_threshold})")
            print(f"   * Media de IoUs de las detecciones correctas: {np.mean(ious_list): .3f}\n")

        return true_positives, false_positives, false_negatives, np.mean(ious_list)

    def _read_class_bboxes_file(self, file_path):
        with open(file_path) as f:
            gt_bboxes = []
            fileline = f.readlines()
            for line in fileline:
                line = line.split(',')[:-1]
                line[0] = line[0].split('.')[0].split('_')[1]
                for idx in range(1, 5):
                    line[idx] = int(line[idx])
                gt_bboxes.append(line)

        return gt_bboxes

    def _read_bboxes_file(self, file_path):
        """
        Read bounding boxes from file. File lines are written like:
         "030000_S29.png,71,27,105,104," or like "030000_S29.png,71,27,105,104" (Glyphdataset style)

        :param file_path: Location of the file containing the bounding boxes.
        :return: Read bounding boxes.
        """
        with open(file_path) as f:
            gt_bboxes = f.readlines()
            gt_bboxes = [box.split(',')[1:] for box in gt_bboxes]
            if len(gt_bboxes[0]) > 4:
                gt_bboxes = [box[:-1] for box in gt_bboxes if len(box) > 4]
            gt_bboxes = [[int(coord) for coord in box] for box in gt_bboxes]

        return gt_bboxes

    def _intersection_over_union(self, bbox1, bbox2):
        """
        Computes the intersection over union between two bounding boxes.
        :param bbox1:
        :param bbox2:
        :return: IoU
        """
        xA = max(bbox1[0], bbox2[0])
        yA = max(bbox1[1], bbox2[1])
        xB = min(bbox1[2], bbox2[2])
        yB = min(bbox1[3], bbox2[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)

        boxAArea = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        boxBArea = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def get_bboxes_from_YOLO_inference(self, model, image_path):
        """
        :param model: YOLO model
        :param image_path: Path of the image to infer.
        :return: Inferred bounding boxes.
        """
        # model = YOLO(model_path)
        inference = model(image_path)[0]
        bboxes = inference.boxes.xyxy.tolist()
        bboxes = [[int(coord) for coord in bbox] for bbox in bboxes]

        return bboxes
