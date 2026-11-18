<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Storage;

class AnimalDetectionController extends Controller
{
    /**
     * Tampilkan Halaman Utama Dashboard Deteksi Hewan CNN & Sensor
     */
    public function index()
    {
        $cameras = DB::table('sensor_cameras')->orderBy('camera_id', 'asc')->get();
        $classes = DB::table('animal_classes')->orderBy('class_id', 'asc')->get();
        $models = DB::table('ai_models')->orderBy('model_id', 'asc')->get();
        $logs = DB::table('detection_logs')->orderBy('log_id', 'desc')->limit(50)->get();

        $stats = [
            'total_detections' => DB::table('detection_logs')->count(),
            'avg_latency' => round(DB::table('detection_logs')->avg('latency_ms') ?? 18.4, 1),
            'avg_confidence' => round(DB::table('detection_logs')->avg('confidence_score') ?? 95.6, 1),
            'active_cameras' => $cameras->where('status', 'ACTIVE')->count(),
            'kucing' => DB::table('detection_logs')->where('detected_class', 'Kucing')->count(),
            'kelinci' => DB::table('detection_logs')->where('detected_class', 'Kelinci')->count(),
            'burung' => DB::table('detection_logs')->where('detected_class', 'Burung')->count(),
            'ayam' => DB::table('detection_logs')->where('detected_class', 'Ayam')->count(),
        ];

        return view('animal_detection.index', compact('cameras', 'classes', 'models', 'logs', 'stats'));
    }

    /**
     * API Endpoint: Simpan Log Deteksi Baru (dari Python Camera Sensor / Roboflow)
     */
    public function storeLog(Request $request)
    {
        $validated = $request->validate([
            'camera_id' => 'required|string',
            'detected_class' => 'required|string|in:Kucing,Kelinci,Burung,Ayam',
            'confidence_score' => 'required|numeric|min:0|max:100',
            'bbox_x' => 'nullable|integer',
            'bbox_y' => 'nullable|integer',
            'bbox_w' => 'nullable|integer',
            'bbox_h' => 'nullable|integer',
            'image_evidence_path' => 'nullable|string',
            'latency_ms' => 'required|numeric',
            'segmentation_method' => 'nullable|string'
        ]);

        $id = DB::table('detection_logs')->insertGetId([
            'camera_id' => $validated['camera_id'],
            'detected_class' => $validated['detected_class'],
            'confidence_score' => $validated['confidence_score'],
            'bbox_x' => $validated['bbox_x'] ?? 0,
            'bbox_y' => $validated['bbox_y'] ?? 0,
            'bbox_w' => $validated['bbox_w'] ?? 0,
            'bbox_h' => $validated['bbox_h'] ?? 0,
            'image_evidence_path' => $validated['image_evidence_path'] ?? 'captured_evidence/default.jpg',
            'latency_ms' => $validated['latency_ms'],
            'segmentation_method' => $validated['segmentation_method'] ?? 'K-Means L*a*b*',
            'timestamp' => now()
        ]);

        return response()->json([
            'success' => true,
            'log_id' => $id,
            'message' => 'Log deteksi berhasil disimpan ke database'
        ], 201);
    }

    /**
     * API Endpoint: Ambil Log Telemetri Real-Time dalam Format JSON
     */
    public function getLogs(Request $request)
    {
        $query = DB::table('detection_logs');

        if ($request->has('class') && $request->class !== 'ALL') {
            $query->where('detected_class', $request->class);
        }
        if ($request->has('camera_id') && $request->camera_id !== 'ALL') {
            $query->where('camera_id', $request->camera_id);
        }

        $logs = $query->orderBy('log_id', 'desc')->limit(100)->get();

        return response()->json([
            'success' => true,
            'data' => $logs
        ]);
    }
}
