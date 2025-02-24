#!/bin/bash
python run_PIPNet.py --seq_name $1
python run_facer.py --seq_name $1
python run_matting_images.py --seq_name $1
python run_MICA.py --seq_name $1
python run_metrical_tracker.py --seq_name $1 $2 --no-intrinsics_provided
