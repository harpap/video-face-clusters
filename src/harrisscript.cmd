@ECHO OFF
setlocal enableextensions EnableDelayedExpansion

IF NOT "%~1"=="trainTest" (
	IF NOT "%~1"=="testOnly" (
		echo type trainTest or testOnly
		GOTO End
	)
)

cd C:\Users\user\Documents\GitHub\facenet-private
IF "%~1"=="trainTest" (
	python src/align/align_dataset_mtcnn.py ~/datasets/lfw/small_raw ~/datasets/lfw/small_lfw_mtcnnpy_160 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25
	python src/classifier.py TRAIN ~/datasets/lfw/small_lfw_mtcnnpy_160 ~/models\facenet\20170512-110547/20170512-110547.pb ~/models/lfw_classifier.pkl --batch_size 1000
	python src/align/align_dataset_mtcnn.py ~/datasets/lfw/test_my_img ~/datasets/lfw/test_my_img_160 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25
	python src/classifier.py CLASSIFY ~/datasets/lfw/test_my_img_160 ~/models\facenet\20170512-110547/20170512-110547.pb ~/models/lfw_classifier.pkl --batch_size 1000
)
IF "%~1"=="testOnly" (
	python src/align/align_dataset_mtcnn.py ~/datasets/lfw/test_my_img ~/datasets/lfw/test_my_img_160 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25
	python src/classifier.py CLASSIFY ~/datasets/lfw/test_my_img_160 ~/models\facenet\20170512-110547/20170512-110547.pb ~/models/lfw_classifier.pkl --batch_size 1000
)


:End