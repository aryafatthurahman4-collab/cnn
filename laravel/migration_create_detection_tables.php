<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        // 1. Sensor Cameras
        Schema::create('sensor_cameras', function (Blueprint $table) {
            $table->string('camera_id', 50)->primary();
            $table->string('location_name', 100);
            $table->string('resolution', 20)->default('1920x1080');
            $table->integer('fps')->default(30);
            $table->string('status', 20)->default('ACTIVE');
            $table->string('stream_url', 255)->default('rtsp://192.168.1.101/live');
            $table->string('model_engine', 100)->default('MobileNetV3 + K-Means L*a*b*');
            $table->timestamp('created_at')->useCurrent();
        });

        // 2. Animal Classes
        Schema::create('animal_classes', function (Blueprint $table) {
            $table->id('class_id');
            $table->string('class_name', 50)->unique();
            $table->string('scientific_name', 100)->nullable();
            $table->string('icon', 50)->default('paw');
            $table->string('risk_level', 20)->default('LOW');
            $table->text('description')->nullable();
        });

        // 3. AI Models Registry
        Schema::create('ai_models', function (Blueprint $table) {
            $table->string('model_id', 50)->primary();
            $table->string('model_name', 100);
            $table->string('framework', 50);
            $table->string('backbone', 100);
            $table->string('input_shape', 50)->default('224x224x3');
            $table->float('accuracy')->default(0);
            $table->float('loss')->default(0);
            $table->float('map50')->default(0);
            $table->float('latency_ms')->default(0);
            $table->string('weights_path', 255)->nullable();
            $table->string('status', 20)->default('READY');
        });

        // 4. Detection Telemetry Logs
        Schema::create('detection_logs', function (Blueprint $table) {
            $table->id('log_id');
            $table->string('camera_id', 50);
            $table->string('detected_class', 50);
            $table->float('confidence_score');
            $table->integer('bbox_x')->default(0);
            $table->integer('bbox_y')->default(0);
            $table->integer('bbox_w')->default(0);
            $table->integer('bbox_h')->default(0);
            $table->string('image_evidence_path', 255);
            $table->float('latency_ms');
            $table->string('segmentation_method', 50)->default('K-Means L*a*b*');
            $table->timestamp('timestamp')->useCurrent();

            $table->foreign('camera_id')->references('camera_id')->on('sensor_cameras')->onDelete('cascade');
        });

        // 5. Roboflow Workflows
        Schema::create('roboflow_workflows', function (Blueprint $table) {
            $table->string('workflow_id', 100)->primary();
            $table->string('workspace', 100);
            $table->string('workflow_url', 255);
            $table->string('project_name', 100);
            $table->string('endpoint_version', 20)->default('v2');
            $table->text('classes_supported');
            $table->string('status', 20)->default('CONNECTED');
            $table->timestamp('last_sync')->useCurrent();
        });

        // 6. Ultralytics Configs
        Schema::create('ultralytics_configs', function (Blueprint $table) {
            $table->string('config_id', 50)->primary();
            $table->string('github_repo', 255)->default('https://github.com/ultralytics/ultralytics');
            $table->string('explore_url', 255)->default('https://platform.ultralytics.com/explore');
            $table->string('model_variant', 50)->default('YOLOv8n');
            $table->string('task', 50)->default('detect');
            $table->float('conf_threshold')->default(0.50);
            $table->float('iou_threshold')->default(0.45);
            $table->string('status', 20)->default('ACTIVE');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('detection_logs');
        Schema::dropIfExists('sensor_cameras');
        Schema::dropIfExists('animal_classes');
        Schema::dropIfExists('ai_models');
        Schema::dropIfExists('roboflow_workflows');
        Schema::dropIfExists('ultralytics_configs');
    }
};
