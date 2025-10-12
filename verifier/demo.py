from verify import verify_video_condition

# videos_path = "/root/machine-gun-baby/diffusiontraining/DiffSynth-Studio/diego/data/train"
fail_video_path = "fail.mp4"
success_video_path = "success.mp4"
condition = "the robot picks up the red cube and places it on the blue cube"

fail_result = verify_video_condition(fail_video_path, condition)
success_result = verify_video_condition(success_video_path, condition)
print(f"failure video eval result: {fail_result.condition_met} for {fail_video_path}")
print(f"success video eval result: {success_result.condition_met} for {success_video_path}")