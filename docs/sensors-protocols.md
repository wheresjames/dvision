# Common Sensor Types for Aerial Drones and Common Protocols

This reference separates **physical/device interfaces** used to connect sensors to flight computers from **MAVLink messages/microservices** used to expose sensor data over the vehicle network.

> **Key distinction:** MAVLink often carries the interpreted measurement, not the sensor's native stream. For example, a LiDAR may connect over UART or DroneCAN but appear in MAVLink as `DISTANCE_SENSOR` or `OBSTACLE_DISTANCE`. A vision camera may deliver images over USB or Ethernet while sending only odometry, control, or metadata over MAVLink.

## Common Aerial-Drone Sensors and MAVLink Support

| Sensor type | Typical purpose | Common hardware / device protocols | Typical MAVLink representation | MAVLink support |
|---|---|---|---|---|
| **Accelerometer** | Linear acceleration, attitude estimation | SPI, I²C, CAN/DroneCAN | `RAW_IMU`, `SCALED_IMU`, `HIGHRES_IMU` | Native |
| **Gyroscope** | Angular velocity, attitude stabilization | SPI, I²C, CAN/DroneCAN | `RAW_IMU`, `SCALED_IMU`, `HIGHRES_IMU` | Native |
| **Magnetometer / compass** | Heading / magnetic field | I²C, SPI, CAN/DroneCAN | `RAW_IMU`, `SCALED_IMU`, `HIGHRES_IMU` | Native |
| **Barometer** | Pressure altitude | I²C, SPI, CAN/DroneCAN | `SCALED_PRESSURE`, `SCALED_PRESSURE2/3`, `HIGHRES_IMU` | Native |
| **Differential-pressure / airspeed** | Fixed-wing airspeed | I²C, analog, CAN/DroneCAN | `AIRSPEED`, `SCALED_PRESSURE` | Native |
| **GNSS / GPS** | Position, velocity, time | UART, USB, CAN/DroneCAN; NMEA, UBX, proprietary binary | `GPS_RAW_INT`, `GPS2_RAW`, `GLOBAL_POSITION_INT` | Native |
| **RTK GNSS** | Centimetre-level positioning | UART, USB, CAN; RTCM3, NTRIP/IP | GPS messages + `GPS_RTCM_DATA` | Native |
| **Dual-antenna GNSS heading** | Heading without magnetometer | UART, CAN/DroneCAN | GPS yaw fields / position messages | Native |
| **Optical-flow sensor** | Velocity estimation near ground / GPS-denied flight | SPI, UART, I²C, CAN/DroneCAN | `OPTICAL_FLOW_RAD` | Native |
| **Visual odometry / VIO** | GPS-denied pose and velocity | USB, MIPI CSI-2, Ethernet; ROS/ROS 2 internally | `ODOMETRY`, `VISION_POSITION_ESTIMATE` | Processed data |
| **UWB positioning** | Indoor/local positioning | UART, SPI, USB, CAN, Ethernet | `ODOMETRY`, `VISION_POSITION_ESTIMATE`, sometimes other positioning messages | Usually processed |
| **Ultrasonic sonar** | Low-altitude ranging / obstacle detection | GPIO pulse, UART, I²C, analog | `DISTANCE_SENSOR` with ultrasound sensor type | Native |
| **Infrared rangefinder** | Short-range distance measurement | Analog, I²C, UART | `DISTANCE_SENSOR` with infrared sensor type | Native |
| **IR / laser ToF sensor** | Altitude / proximity | I²C, UART, CAN | `DISTANCE_SENSOR` | Native |
| **Single-beam LiDAR** | Terrain height, landing, forward ranging | UART, I²C, CAN/DroneCAN, analog/PWM | `DISTANCE_SENSOR` with laser sensor type | Native |
| **2D / 360° LiDAR** | Obstacle avoidance | UART, USB, CAN, Ethernet | `OBSTACLE_DISTANCE`; sometimes multiple `DISTANCE_SENSOR` messages | Reduced representation |
| **3D LiDAR** | Mapping, SLAM, obstacle avoidance | Ethernet/UDP, USB, CAN | `OBSTACLE_DISTANCE` / `ODOMETRY` after processing | Processed |
| **Radar altimeter** | Altitude over difficult surfaces | CAN, UART, RS-422/485 | `DISTANCE_SENSOR` with radar sensor type | Native |
| **Radar obstacle sensor** | Long-range obstacle detection | CAN, UART, Ethernet | `OBSTACLE_DISTANCE` / `DISTANCE_SENSOR` | Native / processed |
| **RGB / visible camera** | Inspection, photography, CV | MIPI CSI-2, USB/UVC, Ethernet, RTSP/RTP, HDMI | Camera Protocol, `CAMERA_INFORMATION`, `CAMERA_IMAGE_CAPTURED`, `VIDEO_STREAM_INFORMATION` | Control / metadata |
| **Near-IR / NIR camera** | Vegetation, inspection, night imaging | USB, CSI-2, Ethernet | Camera Protocol / video stream metadata | Partial |
| **Thermal / LWIR camera** | Heat detection, search & rescue, inspection | USB/UVC, Ethernet/RTSP, CSI-2, serial | Camera Protocol + `CAMERA_THERMAL_RANGE` | Specialized support |
| **Stereo camera** | Depth, VIO, SLAM | USB, MIPI CSI-2, Ethernet | `ODOMETRY`, `OBSTACLE_DISTANCE`, camera messages | Processed |
| **Depth / ToF camera** | Depth maps / obstacle avoidance | USB, CSI-2, Ethernet | `OBSTACLE_DISTANCE`, `DISTANCE_SENSOR`, `ODOMETRY` | Processed |
| **Multispectral camera** | Agriculture, NDVI, environmental surveys | USB, Ethernet, serial/GPIO trigger | Camera Protocol for control; imagery separate | Partial |
| **Hyperspectral camera** | Scientific/agricultural/mineral sensing | USB 3, Ethernet, CameraLink/GigE | Usually no standardized spectral MAVLink payload | Custom |
| **Event camera** | High-speed vision / navigation | USB, MIPI, Ethernet | Usually processed to `ODOMETRY` | Custom / processed |
| **Battery voltage/current sensor** | Power monitoring | ADC/analog, I²C/SMBus, CAN/DroneCAN | `BATTERY_STATUS`, `BATTERY_INFO`, `POWER_STATUS` | Native |
| **Motor / RPM sensor** | Propeller/engine RPM | GPIO frequency, UART, CAN, ESC telemetry | `RAW_RPM`, ESC status messages | Native |
| **Temperature / humidity** | Environmental / electronics monitoring | I²C, SPI, UART, CAN | `HYGROMETER_SENSOR`; temperature also embedded in several messages | Native |
| **Gas / air-quality sensor** | Methane, CO₂, VOC, pollutants | Analog, I²C, UART, RS-485, CAN | Usually custom MAVLink dialect or `NAMED_VALUE_*` for simple telemetry | Custom |
| **Radiation sensor** | Gamma/radiological survey | UART, SPI, USB, CAN | Usually custom dialect / companion computer | Custom |
| **ADS-B receiver** | Cooperative aircraft detection | UART, USB, Ethernet | `ADSB_VEHICLE`, `COLLISION` | Native |

## Common Sensor Connection Protocols

| Protocol/interface | Typical use on drones | Typical sensors | Comments |
|---|---|---|---|
| **SPI** | High-rate, short-distance onboard connection | IMU, gyro, accelerometer, barometer | Fast and deterministic; normally internal to flight controller |
| **I²C** | Low/medium bandwidth peripherals | Compass, barometer, airspeed, rangefinder | Simple two-wire multidrop bus; generally kept short |
| **UART / TTL serial** | Very common external sensor interface | GNSS, LiDAR, radar, telemetry | Often carries a vendor-specific binary/ASCII protocol |
| **RS-232** | Industrial serial equipment | GNSS, payload instruments | Different electrical levels from TTL UART |
| **RS-422 / RS-485** | Long cable / noise-resistant serial | Industrial LiDAR, radar, environmental sensors | Differential signalling; useful on larger aircraft |
| **CAN** | Robust distributed avionics network | GNSS, IMU, rangefinder, ESC, power sensor | Useful for larger UAVs |
| **DroneCAN** | UAV-specific protocol over CAN | GNSS, compass, airspeed, IMU, rangefinder, optical flow, power | Widely supported by PX4 and ArduPilot |
| **Cyphal / UAVCAN v1** | Newer CAN/CAN-FD avionics architecture | Sensors and actuators | Newer protocol family; adoption varies |
| **USB** | High-bandwidth companion-computer peripherals | RGB/depth/thermal cameras, LiDAR | Usually goes to companion computer rather than flight controller |
| **USB UVC** | Standard USB video | RGB/thermal cameras | Standard video-camera interface |
| **MIPI CSI-2** | Direct camera-to-compute interface | RGB, thermal, stereo cameras | High bandwidth, short cable |
| **Ethernet** | High-bandwidth payloads | 3D LiDAR, cameras, radar | Increasingly common on larger UAVs |
| **UDP/TCP over IP** | Sensor data transport | LiDAR, radar, cameras, companion computers | Common over Ethernet |
| **RTSP/RTP** | Video streaming | RGB/IR/thermal cameras | MAVLink may advertise/control the stream while video travels separately |
| **GigE Vision** | Industrial machine-vision cameras | RGB, NIR, thermal, hyperspectral | Common in industrial payloads |
| **NMEA 0183** | GNSS data format | GPS/GNSS | Normally transported over UART |
| **UBX** | u-blox GNSS binary protocol | GNSS/RTK receivers | Normally UART or USB |
| **RTCM 3.x** | GNSS correction data | RTK GNSS | Can be carried through MAVLink `GPS_RTCM_DATA` |
| **SMBus** | Smart battery communications | Batteries / battery monitors | Closely related to I²C |
| **Analog voltage / ADC** | Simple sensor output | Airspeed, sonar, current sensor | Low complexity but susceptible to noise |
| **PWM / pulse-width / pulse-time** | Distance/RPM/trigger signals | Rangefinders, tachometers | Simple interface |
| **GPIO trigger** | Camera synchronization | RGB, multispectral, thermal | Often used with PPS for accurate geotagging |
| **PPS** | Precise timing | GNSS, cameras, LiDAR | Important for sensor synchronization |
| **ROS / ROS 2** | Companion-computer software integration | Cameras, LiDAR, SLAM, radar | Software middleware rather than an electrical sensor interface |
| **MAVLink** | Flight computer ↔ payload ↔ companion ↔ GCS | Processed measurements and device control | Usually not the sensor's lowest-level bus |

## Particularly Useful MAVLink Sensor Messages

| MAVLink message / service | Sensor/data class |
|---|---|
| `HIGHRES_IMU` | Accelerometer, gyro, magnetometer, pressure, temperature |
| `RAW_IMU` / `SCALED_IMU*` | IMU |
| `SCALED_PRESSURE*` | Barometer / differential pressure |
| `AIRSPEED` | Airspeed |
| `GPS_RAW_INT` / `GPS2_RAW` | GNSS |
| `GPS_RTCM_DATA` | RTK correction injection |
| `OPTICAL_FLOW_RAD` | Optical flow |
| `DISTANCE_SENSOR` | LiDAR, ultrasonic, IR, radar |
| `OBSTACLE_DISTANCE` | 2D/360° proximity data |
| `ODOMETRY` | VIO, SLAM, external positioning |
| `VISION_POSITION_ESTIMATE` | Vision position/attitude |
| `CAMERA_INFORMATION` | Camera capabilities |
| `CAMERA_IMAGE_CAPTURED` | Image/geotag metadata |
| `VIDEO_STREAM_INFORMATION` | Video stream description |
| `CAMERA_THERMAL_RANGE` | Thermal camera temperature range |
| `BATTERY_STATUS` | Battery measurements |
| `RAW_RPM` | RPM/tachometer |
| `HYGROMETER_SENSOR` | Temperature/humidity |
| `ADSB_VEHICLE` | ADS-B traffic |

## Typical Integration Pattern

A practical UAV architecture often uses:

- **SPI** for internal high-rate IMUs.
- **DroneCAN/CAN** for distributed flight-critical sensors and smart peripherals.
- **UART** for GNSS, simple LiDAR/rangefinders, and some radar units.
- **Ethernet, USB, or MIPI CSI-2** for high-bandwidth perception sensors such as cameras and 3D LiDAR.
- **MAVLink** as the normalized vehicle-level interface among the autopilot, companion computer, payloads, and ground-control station.

### Cameras and MAVLink

MAVLink generally does **not** carry the primary live video stream. A typical architecture is:

```text
Camera
  └─ USB / MIPI CSI-2 / Ethernet
       └─ Companion computer or video encoder
            └─ RTSP / RTP video stream
```

while control and metadata follow a separate path:

```text
Camera / payload controller
  ↕
MAVLink
  ↕
Autopilot / companion computer / ground-control station
```

MAVLink can therefore handle camera discovery, triggering, capture status, settings, tracking, geotag metadata, and video-stream information while the actual video is transported separately.

## Reference Documentation

- MAVLink Common Message Set: https://mavlink.io/en/messages/common.html
- MAVLink Camera Protocol: https://mavlink.io/en/services/camera.html
- MAVLink Image Transmission: https://mavlink.io/en/services/image_transmission.html
- PX4 Sensor Bus Documentation: https://docs.px4.io/main/en/sensor_bus/
- PX4 DroneCAN Documentation: https://docs.px4.io/main/en/dronecan/
- ArduPilot Sensor Drivers: https://ardupilot.org/dev/docs/code-overview-sensor-drivers.html
