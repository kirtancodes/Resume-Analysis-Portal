from .video_analysis import (
    extract_audio_from_video,
    generate_transcription,
    generate_summary,
    analyze_performance
)

def process_video(video_path, mongo):
    # Extract audio and process video
    audio_path = extract_audio_from_video(video_path)
    transcription = generate_transcription(audio_path)

    if not transcription:
        return {"error": "Failed to generate transcription."}

    # Perform further analysis
    summary = generate_summary(transcription)
    performance_analysis = analyze_performance(transcription)

    # Save results to MongoDB
    mongo.db.video_transcriptions.insert_one({
        "video_path": video_path,
        "transcription": transcription,
        "summary": summary,
        "performance_analysis": performance_analysis
    })

    return {
        "transcription": transcription,
        "summary": summary,
        "performance_analysis": performance_analysis
    }
