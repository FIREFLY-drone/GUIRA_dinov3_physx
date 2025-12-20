# DJI Cloud API Integration

## Overview

This document provides a quickstart guide for integrating with the DJI Cloud API to enable drone telemetry and video streaming for fire prevention monitoring. The DJI Cloud API allows applications to access real-time drone data, control camera settings, and stream live video feeds.

## Prerequisites

- DJI Developer Account
- Registered DJI application  
- DJI drone with Cloud API support (DJI Air 2S, Mini 3, Mavic 3, etc.)
- Active internet connection for drone and ground station

## Authentication & Setup

### 1. DJI Developer Registration

1. Visit [DJI Developer Portal](https://developer.dji.com/)
2. Create developer account and verify identity
3. Create new application in DJI Cloud API console
4. Obtain application credentials:
   - `APP_ID`: Unique application identifier
   - `APP_KEY`: Application secret key  
   - `APP_LICENSE`: Application license string

### 2. Environment Configuration

Add the following environment variables:
```bash
# DJI Cloud API Credentials
export DJI_APP_ID="your_app_id_here"
export DJI_APP_KEY="your_app_key_here"  
export DJI_APP_LICENSE="your_app_license_here"

# Optional: Regional API endpoints
export DJI_API_ENDPOINT="https://api-gateway.dji.com"  # Global
# export DJI_API_ENDPOINT="https://api-gateway-cn.dji.com"  # China
```

### 3. SDK Installation

```bash
# Install DJI Cloud API SDK
pip install dji-cloud-api-sdk

# Or use our integrated controller
pip install -r requirements.txt
```

## Controller Implementation

The fire prevention system includes a DJI controller module at `src/controller/dji_controller.py`:

### Basic Controller Setup

```python
from src.controller.dji_controller import DJICloudController
import os

# Initialize controller
controller = DJICloudController(
    app_id=os.getenv('DJI_APP_ID'),
    app_key=os.getenv('DJI_APP_KEY'),
    app_license=os.getenv('DJI_APP_LICENSE')
)

# Connect to drone
drone_sn = "1234567890ABCDEF"  # Drone serial number
success = await controller.connect_drone(drone_sn)

if success:
    print("✅ Connected to drone successfully")
else:
    print("❌ Failed to connect to drone")
```

### Authentication Flow

```python
class DJICloudController:
    def __init__(self, app_id: str, app_key: str, app_license: str):
        self.app_id = app_id
        self.app_key = app_key
        self.app_license = app_license
        self.access_token = None
        self.refresh_token = None
        
    async def authenticate(self) -> bool:
        """Authenticate with DJI Cloud API"""
        
        auth_payload = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "app_license": self.app_license,
            "grant_type": "client_credentials"
        }
        
        try:
            response = await self._post('/auth/token', auth_payload)
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                self.refresh_token = token_data.get('refresh_token')
                
                # Schedule token refresh
                expires_in = token_data.get('expires_in', 3600)
                asyncio.create_task(self._refresh_token_periodically(expires_in))
                
                return True
            else:
                logger.error(f"Authentication failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
```

## Telemetry Integration

### MQTT Subscription Setup

```python
import paho.mqtt.client as mqtt
import json

class DJITelemetryHandler:
    def __init__(self, controller: DJICloudController):
        self.controller = controller
        self.mqtt_client = None
        self.telemetry_callback = None
        
    async def start_telemetry_stream(self, drone_sn: str, callback: callable):
        """Start receiving telemetry data via MQTT"""
        
        # Get MQTT connection details from API
        mqtt_config = await self.controller.get_mqtt_config(drone_sn)
        
        # Setup MQTT client
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        self.telemetry_callback = callback
        
        # Connect to MQTT broker
        self.mqtt_client.connect(
            mqtt_config['host'], 
            mqtt_config['port'], 
            keepalive=60
        )
        
        # Subscribe to telemetry topics
        topics = [
            f"dji/{drone_sn}/telemetry/position",
            f"dji/{drone_sn}/telemetry/attitude", 
            f"dji/{drone_sn}/telemetry/battery",
            f"dji/{drone_sn}/telemetry/gimbal"
        ]
        
        for topic in topics:
            self.mqtt_client.subscribe(topic)
            
        self.mqtt_client.loop_start()
        
    def _on_mqtt_message(self, client, userdata, message):
        """Process incoming MQTT telemetry message"""
        try:
            topic = message.topic
            payload = json.loads(message.payload.decode())
            
            # Parse telemetry data
            telemetry_data = self._parse_telemetry(topic, payload)
            
            # Forward to callback
            if self.telemetry_callback:
                self.telemetry_callback(telemetry_data)
                
        except Exception as e:
            logger.error(f"Error processing telemetry: {e}")
```

### Telemetry Data Structure

```python
@dataclass
class DroneTelemetry:
    """Standard telemetry data structure"""
    
    # Position data
    latitude: float
    longitude: float  
    altitude: float          # Meters above takeoff
    altitude_abs: float      # Meters above sea level
    
    # Attitude data
    roll: float              # Degrees
    pitch: float             # Degrees
    yaw: float               # Degrees (0=North)
    
    # Velocity data
    velocity_x: float        # m/s (North-South)
    velocity_y: float        # m/s (East-West)
    velocity_z: float        # m/s (Up-Down)
    
    # Battery data
    battery_percent: int     # 0-100%
    battery_voltage: float   # Volts
    battery_current: float   # Amps
    
    # Gimbal data
    gimbal_roll: float       # Degrees
    gimbal_pitch: float      # Degrees  
    gimbal_yaw: float        # Degrees
    
    # Flight status
    flight_mode: str         # "AUTO", "MANUAL", "RTH", etc.
    gps_signal_level: int    # 0-5 (5 = excellent)
    home_distance: float     # Distance to home point (m)
    
    # Timestamps
    timestamp: datetime
    system_time: float       # Unix timestamp
```

## Video Streaming

### Camera Control

```python
class DJICameraController:
    def __init__(self, controller: DJICloudController):
        self.controller = controller
        
    async def set_camera_mode(self, drone_sn: str, mode: str) -> bool:
        """Set camera mode (photo, video, etc.)"""
        
        payload = {
            "drone_sn": drone_sn,
            "camera_mode": mode,  # "photo", "video", "recording"
            "camera_index": 0     # Primary camera
        }
        
        response = await self.controller._post('/camera/mode', payload)
        return response.status_code == 200
        
    async def set_zoom_level(self, drone_sn: str, zoom_factor: float) -> bool:
        """Set camera zoom level"""
        
        payload = {
            "drone_sn": drone_sn,
            "zoom_factor": zoom_factor,  # 1.0 = no zoom, 2.0 = 2x zoom
            "camera_index": 0
        }
        
        response = await self.controller._post('/camera/zoom', payload)
        return response.status_code == 200
        
    async def set_exposure_settings(self, drone_sn: str, 
                                  iso: int = None, 
                                  shutter_speed: str = None,
                                  aperture: str = None) -> bool:
        """Configure camera exposure settings"""
        
        payload = {
            "drone_sn": drone_sn,
            "camera_index": 0
        }
        
        if iso:
            payload["iso"] = iso  # 100, 200, 400, 800, 1600, 3200
        if shutter_speed:
            payload["shutter_speed"] = shutter_speed  # "1/60", "1/120", etc.
        if aperture:
            payload["aperture"] = aperture  # "f/2.8", "f/4.0", etc.
            
        response = await self.controller._post('/camera/exposure', payload)
        return response.status_code == 200
```

### Live Stream Setup

```python
class DJILiveStreamHandler:
    def __init__(self, controller: DJICloudController):
        self.controller = controller
        self.stream_active = False
        
    async def start_live_stream(self, drone_sn: str, 
                              quality: str = "1080p",
                              frame_rate: int = 30) -> dict:
        """Start live video stream from drone"""
        
        # Request stream initialization  
        stream_config = {
            "drone_sn": drone_sn,
            "camera_index": 0,
            "stream_quality": quality,    # "720p", "1080p", "4K"
            "frame_rate": frame_rate,     # 24, 30, 60 fps
            "stream_type": "rtmp"         # "rtmp", "hls", "webrtc"
        }
        
        response = await self.controller._post('/stream/start', stream_config)
        
        if response.status_code == 200:
            stream_info = response.json()
            
            return {
                "stream_url": stream_info['stream_url'],
                "stream_key": stream_info['stream_key'],
                "rtmp_endpoint": stream_info['rtmp_endpoint'],
                "hls_playlist": stream_info.get('hls_playlist'),
                "session_id": stream_info['session_id']
            }
        else:
            raise Exception(f"Failed to start stream: {response.status_code}")
            
    async def stop_live_stream(self, session_id: str) -> bool:
        """Stop active live stream"""
        
        payload = {"session_id": session_id}
        response = await self.controller._post('/stream/stop', payload)
        
        return response.status_code == 200
```

## Integration with Fire Prevention System

### Real-time Data Pipeline

```python
from src.video_streaming import VideoStreamProcessor
from src.maps_adapter import DronePose, create_fire_overlay

class FirePreventionDJIIntegration:
    def __init__(self):
        self.dji_controller = DJICloudController(...)
        self.stream_processor = VideoStreamProcessor(...)
        self.current_pose = None
        
    async def start_monitoring_mission(self, drone_sn: str):
        """Start complete fire monitoring pipeline"""
        
        # 1. Connect to drone
        await self.dji_controller.connect_drone(drone_sn)
        
        # 2. Configure camera for fire detection
        camera = DJICameraController(self.dji_controller)
        await camera.set_camera_mode(drone_sn, "video")
        await camera.set_exposure_settings(drone_sn, iso=400)  # Good for daylight fire detection
        
        # 3. Start telemetry stream
        telemetry = DJITelemetryHandler(self.dji_controller)
        await telemetry.start_telemetry_stream(drone_sn, self._telemetry_callback)
        
        # 4. Start video stream  
        stream_handler = DJILiveStreamHandler(self.dji_controller)
        stream_info = await stream_handler.start_live_stream(drone_sn, "1080p", 30)
        
        # 5. Connect video stream to fire detection pipeline
        await self._connect_stream_to_processor(stream_info['rtmp_endpoint'])
        
        print("🔥 Fire monitoring mission started")
        
    def _telemetry_callback(self, telemetry: DroneTelemetry):
        """Process incoming telemetry data"""
        
        # Update current drone pose
        self.current_pose = DronePose(
            latitude=telemetry.latitude,
            longitude=telemetry.longitude,
            altitude=telemetry.altitude_abs,
            roll=telemetry.roll,
            pitch=telemetry.pitch, 
            yaw=telemetry.yaw,
            timestamp=telemetry.system_time
        )
        
        # Log critical telemetry
        if telemetry.battery_percent < 30:
            logger.warning(f"⚠️ Low battery: {telemetry.battery_percent}%")
            
        if telemetry.gps_signal_level < 3:
            logger.warning(f"⚠️ Weak GPS signal: {telemetry.gps_signal_level}/5")
            
    async def _connect_stream_to_processor(self, rtmp_url: str):
        """Connect RTMP stream to fire detection processor"""
        
        # Use OpenCV to capture RTMP stream
        cap = cv2.VideoCapture(rtmp_url)
        
        while self.stream_processor.running:
            ret, frame = cap.read()
            
            if ret:
                # Process frame through fire detection
                success = self.stream_processor.process_frame(
                    frame, 
                    drone_pose=self.current_pose
                )
                
                if success:
                    # Get latest detection results
                    result = self.stream_processor.get_latest_result()
                    
                    if result and result.fire_detections:
                        await self._handle_fire_detection(result)
                        
            await asyncio.sleep(1/30)  # 30 FPS processing
            
    async def _handle_fire_detection(self, detection_result):
        """Handle fire detection alert"""
        
        fire_count = detection_result.fire_detections.get('detection_count', 0)
        
        if fire_count > 0:
            logger.warning(f"🔥 FIRE DETECTED: {fire_count} fire(s) at {self.current_pose.latitude:.6f}, {self.current_pose.longitude:.6f}")
            
            # Send alert to ground station
            await self._send_fire_alert(detection_result)
            
            # Optionally adjust drone behavior
            await self._respond_to_fire_detection()
            
    async def _send_fire_alert(self, detection_result):
        """Send fire detection alert to ground control"""
        
        alert_data = {
            "alert_type": "FIRE_DETECTED",
            "timestamp": detection_result.timestamp,
            "location": {
                "latitude": self.current_pose.latitude,
                "longitude": self.current_pose.longitude,
                "altitude": self.current_pose.altitude
            },
            "detections": detection_result.fire_detections,
            "confidence": max(detection_result.fire_detections.get('scores', [0])),
            "geojson_overlay": detection_result.fire_overlay_geojson
        }
        
        # Send to ground control system
        await self._notify_ground_control(alert_data)
```

## JSBridge Integration

For mobile/tablet ground station apps, DJI provides JSBridge for JavaScript integration:

### JSBridge Setup

```javascript
// Initialize DJI JSBridge
DJI.onSDKLoaded = function() {
    // SDK loaded successfully
    console.log("DJI SDK Loaded");
    
    // Connect to drone
    DJI.connectToDrone({
        success: function(result) {
            console.log("Connected to drone:", result);
            startFireMonitoring();
        },
        error: function(error) {
            console.error("Connection failed:", error);
        }
    });
};

function startFireMonitoring() {
    // Start telemetry stream
    DJI.startTelemetryUpdates({
        onUpdate: function(telemetry) {
            // Send telemetry to fire prevention backend
            sendTelemetryToBackend(telemetry);
        }
    });
    
    // Start camera stream
    DJI.startVideoStreaming({
        quality: "1080p",
        onFrame: function(frameData) {
            // Send frame to fire detection API
            processFrameForFire(frameData);
        }
    });
}

function sendTelemetryToBackend(telemetry) {
    fetch('/api/telemetry/update', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(telemetry)
    });
}
```

## Best Practices & Tips

### 1. Connection Management

- **Implement retry logic** for API connections
- **Monitor connection health** with heartbeat messages  
- **Handle network interruptions** gracefully
- **Use exponential backoff** for failed requests

### 2. Battery & Safety

- **Monitor battery levels** continuously (alert at 30%, RTH at 20%)
- **Implement geofencing** to prevent restricted area entry
- **Set return-to-home triggers** for low battery or signal loss
- **Log all flight data** for post-mission analysis

### 3. Performance Optimization

- **Limit telemetry frequency** to 1-5 Hz (avoid API rate limits)
- **Use appropriate video quality** based on network bandwidth
- **Implement frame dropping** during network congestion  
- **Buffer critical data** locally during connection drops

### 4. Error Handling

```python
class DJIErrorHandler:
    @staticmethod
    async def handle_api_error(response, operation: str):
        """Centralized API error handling"""
        
        if response.status_code == 401:
            logger.error("Authentication expired, refreshing token...")
            # Trigger token refresh
            
        elif response.status_code == 429:
            logger.warning("Rate limit exceeded, backing off...")
            await asyncio.sleep(60)  # Wait 1 minute
            
        elif response.status_code == 503:
            logger.warning("DJI service unavailable, retrying...")
            await asyncio.sleep(30)
            
        else:
            logger.error(f"API error in {operation}: {response.status_code} - {response.text}")
```

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Verify APP_ID, APP_KEY, and APP_LICENSE are correct
   - Check if application is approved in DJI Developer Portal
   - Ensure drone is connected to internet

2. **Stream Quality Issues**  
   - Reduce video quality if bandwidth is limited
   - Check RTMP/HLS endpoint accessibility
   - Verify firewall settings allow streaming ports

3. **Telemetry Gaps**
   - Check MQTT broker connection stability
   - Verify topic subscription patterns
   - Monitor for message queue overflow

4. **GPS/Positioning Problems**
   - Ensure clear sky view for GPS reception
   - Wait for GPS signal strength ≥ 4 before takeoff
   - Check for magnetic interference affecting compass

### Debug Commands

```bash
# Test DJI API connectivity
curl -X POST https://api-gateway.dji.com/auth/token \
  -H "Content-Type: application/json" \
  -d '{"app_id":"YOUR_APP_ID","app_key":"YOUR_APP_KEY","app_license":"YOUR_LICENSE","grant_type":"client_credentials"}'

# Test MQTT connection  
mosquitto_sub -h dji-mqtt-broker.com -p 1883 -t "dji/+/telemetry/+"

# Test video stream
ffplay rtmp://dji-stream-server.com/live/YOUR_STREAM_KEY
```

## Security Considerations

- **Store credentials securely** (use environment variables or secret management)
- **Implement access controls** for drone operations
- **Encrypt data transmission** when possible
- **Log security events** (authentication, authorization)
- **Regular security audits** of API integration code

## Support & Resources

- **DJI Developer Portal**: https://developer.dji.com/
- **API Documentation**: https://developer.dji.com/cloud-api/docs/
- **Community Forum**: https://forum.dji.com/forum-139-1.html  
- **Technical Support**: Contact DJI Developer Support

For issues specific to fire prevention integration, check the project documentation or create an issue in the repository.
