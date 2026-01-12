import unittest
from unittest.mock import patch, MagicMock
import cv2 as cv
import src.mediapipe_intro_in.mediapipe_hands  # Run the script

class TestMediaPipeHands(unittest.TestCase):
    @patch('mediapipe_intro_in.mediapipe_hands.cv.VideoCapture')
    @patch('mediapipe_intro_in.mediapipe_hands.mp_hands.Hands')
    def test_hand_detection_loop(self, mock_hands_class, mock_videocapture_class):
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.side_effect = [True, False]  # Loop runs once
        mock_cap.read.return_value = (True, MagicMock())
        mock_videocapture_class.return_value = mock_cap

        # Mock Hands
        mock_hands = MagicMock()
        mock_hands.process.return_value = MagicMock(multi_hand_landmarks=[MagicMock()])
        mock_hands_class.return_value = mock_hands

        # Patch cv2 functions
        with patch('mediapipe_intro_in.mediapipe_hands.cv.cvtColor'), \
             patch('mediapipe_intro_in.mediapipe_hands.cv.imshow'), \
             patch('mediapipe_intro_in.mediapipe_hands.cv.waitKey', return_value=27), \
             patch('mediapipe_intro_in.mediapipe_hands.cv.destroyAllWindows'), \
             patch('mediapipe_intro_in.mediapipe_hands.drawing_utils.draw_landmarks'):


            # Assertions
            mock_cap.read.assert_called()
            mock_hands.process.assert_called()
            mock_videocapture_class.assert_called_with(0)

if __name__ == '__main__':
    unittest.main()