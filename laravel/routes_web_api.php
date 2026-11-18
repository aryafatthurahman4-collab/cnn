<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\AnimalDetectionController;

/*
|--------------------------------------------------------------------------
| Web & API Routes untuk AI Animal Vision & Telemetri Sensor
| Tambahkan rute ini ke routes/web.php atau routes/api.php
|--------------------------------------------------------------------------
*/

Route::prefix('animal-detector')->group(function () {
    // Web Dashboard
    Route::get('/', [AnimalDetectionController::class, 'index'])->name('animal.detector.index');

    // API Telemetri & Inference
    Route::get('/api/logs', [AnimalDetectionController::class, 'getLogs'])->name('animal.detector.logs');
    Route::post('/api/logs', [AnimalDetectionController::class, 'storeLog'])->name('animal.detector.store');
});
