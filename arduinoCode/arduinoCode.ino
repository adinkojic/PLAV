#include <Arduino.h>

float altitude_msl = 10013;
float equivalent_airspeed = 287.8;
float angle_of_attack = 2.6538;
float angle_of_sideslip = 0.0;
float euler_angle_roll = 0.0;
float euler_angle_pitch = 2.6538;
float euler_angle_yaw = 45.0;
float body_angular_rate_roll = 0.0;
float body_angular_rate_pitch = 0.0;
float body_angular_rate_yaw = 0.0;
float sim_time = 0.0;
bool staleData = true;

float trim_rdr   = 0.0;
float trim_ail   = 0.0;
float trim_el    = -3.2410;
float trim_power = 13.9019;

float pilot_control_throttle = 0.0;
float pilot_control_long = 0.0;
float pilot_control_lat = 0.0;
float pilot_control_yaw = 0.0;

float stability_augmentation_on_disc = 1.0;
float autopilot_on_disc = 1.0;
float equivalent_airspeed_command = 287.8;
float altitude_msl_command =10013.0;
float lateral_deviation_error = 0.0;
float true_base_course_command = 45.0;

float float_scaling = 21474.83648;

float controlDeflections[4] = {};

// Wraps an angle value to lie within [min, max]
double wrapAngle(double value, double min, double max) {
    double period = max - min;
    double result = value;
    while (result > max)  result -= period;
    while (result < min)  result += period;
    return result;
}

double clamp(double value, double min, double max){
  if(value > max){
    return max;
  }
  if(value < min){
    return min;
  }
  return value;
}


int combineBytes(uint8_t rawData[44], int offset){
  int result = (rawData[3+offset] << 24) | (rawData[2+offset] << 16) | (rawData[1+offset] << 8) | (rawData[0+offset]);
  return result;
}



void rawDataToEnviroment(uint8_t rawData[44]){
  altitude_msl = combineBytes(rawData, 0) / float_scaling ;
  equivalent_airspeed = combineBytes(rawData, 4) / float_scaling ;
  angle_of_attack = combineBytes(rawData, 8) / float_scaling ;
  angle_of_sideslip = combineBytes(rawData, 12) / float_scaling ;
  euler_angle_roll = combineBytes(rawData, 16) / float_scaling ;
  euler_angle_pitch = combineBytes(rawData, 20) / float_scaling ;
  euler_angle_yaw = combineBytes(rawData, 24) / float_scaling ;
  body_angular_rate_roll = combineBytes(rawData, 28) / float_scaling ;
  body_angular_rate_pitch = combineBytes(rawData, 32) / float_scaling ;
  body_angular_rate_yaw = combineBytes(rawData, 36) / float_scaling ;
  sim_time = combineBytes(rawData, 40) / float_scaling ;

  staleData = false;
}

void rawDataToCommands(uint8_t rawData[17]){
  equivalent_airspeed_command = combineBytes(rawData, 0) / float_scaling ;
  altitude_msl_command = combineBytes(rawData, 4) / float_scaling ;
  lateral_deviation_error = combineBytes(rawData, 8) / float_scaling ;
  true_base_course_command = combineBytes(rawData, 12) / float_scaling ;
  stability_augmentation_on_disc = (rawData[16] & 1) ? 1.0 : 0.0;
  autopilot_on_disc = (rawData[16] & 2) ? 1.0 : 0.0;
}

// Core LQR‑based control computation
int computeControl(float result[4]) {
  // Nominal values & gains
  const double design_keas   = 287.8088596053291;
  const double design_aoa    = 2.653813535191715;
  const double design_pitch  = 2.653813535191715;
  const double gain_alt      = -0.05;
  const double gain_lat      = -0.01;
  const double gain_track    = -10.0;

  // Mode flags
  bool ap_on  = autopilot_on_disc > 0.5;
  bool sas_on = stability_augmentation_on_disc > 0.5;
  bool fsas   = ap_on || sas_on;

  
  // Switched setpoints
  double keas_sp = ap_on ? equivalent_airspeed_command : design_keas;
  double pitch_delta = clamp(gain_alt*(altitude_msl - altitude_msl_command), -5.0, 5.0);
  double pitch_sp = pitch_delta + design_pitch;
  double theta_sp = ap_on ? pitch_sp : design_pitch;

  // Disturbances
  double dv = equivalent_airspeed - keas_sp;
  double da = angle_of_attack - design_aoa;
  double dt = euler_angle_pitch - theta_sp;

  // Course tracking
  double course_corr = clamp(lateral_deviation_error * gain_lat, -30.0, 30.0);
  double course_sp   = true_base_course_command + course_corr;
  double track_err   = wrapAngle(euler_angle_yaw + angle_of_sideslip - course_sp, -180.0, 180.0);
  double bank_sp     = clamp(gain_track * track_err, -30.0, 30.0);
  double phi_sp      = ap_on ? bank_sp : 0.0;
  double dr          = euler_angle_roll - phi_sp;

  // LQR gain matrices
  static const double longK[2][4] = {
      {-0.063009074230494,  0.113230403179271, 10.113432224566077, 3.154983341632913},
      { 0.997260602961658, -0.025467711176391,  1.213308488207827, 0.208744369535208}
  };
  static const double latK[2][4] = {
      { 3.078043941515770,  0.032365863044163,  4.557858908828332,  0.589443156647647},
      {-0.705817452754520, -0.256362860634868, -1.073666149713151,  0.822114635953878}
  };

  double x_long[4] = {dv, da, body_angular_rate_pitch, dt};
  double x_lat[4]  = {dr, angle_of_sideslip, body_angular_rate_roll, body_angular_rate_yaw};

  double u_longit[2] = {0.0, 0.0};
  double u_latit[2]  = {0.0, 0.0};
  for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 4; ++j) {
          u_longit[i] -= longK[i][j] * x_long[j];
          u_latit[i] -= latK[i][j] * x_lat[j];
      }
  }

  // Pilot plus trim
  double pilot_long[2] = {0.0,0.0};
  double pilot_lat[2] = {0.0,0.0};

  if(ap_on){
    pilot_long[0] = pilot_control_long;
    pilot_long[1] = pilot_control_throttle;
    pilot_lat[0] = pilot_control_lat;
    pilot_lat[1] = pilot_control_yaw;
  }

  double trim_long[2] = {trim_el/25.0, trim_power/100.0};

  double out_long[2];
  double out_lat[2];
  if (fsas) {
      out_long[0] = u_longit[0] + pilot_long[0] + trim_long[0];
      out_long[1] = u_longit[1] + pilot_long[1] + trim_long[1];
      out_lat[0] = u_latit[0] + pilot_lat[0];
      out_lat[1] = u_latit[1] + pilot_lat[1];
  } else {
      out_long[0] = pilot_long[0] + trim_long[0];
      out_long[1] = pilot_long[1] + trim_long[1];
      out_lat[0] = pilot_lat[0];
      out_lat[1] = pilot_lat[1];
  }

  // Saturation & final deflections
  out_long[0] = clamp(out_long[0], -1.0, 1.0);
  out_long[1] = clamp(out_long[1],  0.0, 1.0);
  out_lat[0] = clamp(out_lat[0], -1.0, 1.0);
  out_lat[1] = clamp(out_lat[1], -1.0, 1.0);

  double aileron  = out_lat[0] * -21.5;
  double rudder   = out_lat[1] * -30.0 + aileron * 0.008;
  double elevator = out_long[0] * -25.0;
  double throttle = out_long[1] * 100.0;

  result[0] =  aileron;
  result[1] =  rudder;
  result[2] =  elevator;
  result[3] =  throttle;

  return 0;
}

void controlResponseBytes(uint8_t reponse[16]){


  int aileron = (int) (controlDeflections[0] * float_scaling);
  int rudder = (int) (controlDeflections[1] * float_scaling);
  int elevator = controlDeflections[2] * float_scaling;
  int throttle = controlDeflections[3] * float_scaling;

  //swap from {ail, rud, el, pwr} to {rud, eil, el, pwr}
  reponse[3] = (uint8_t) ((rudder & 0xFF000000) >> 24);
  reponse[2] = (uint8_t) ((rudder & 0x00FF0000) >> 16);
  reponse[1]  = (uint8_t) ((rudder & 0x0000FF00) >> 8);
  reponse[0]  = (uint8_t) (rudder  & 0x000000FF);

  reponse[7] = (uint8_t) ((aileron & 0xFF000000) >> 24);
  reponse[6] = (uint8_t) ((aileron & 0x00FF0000) >> 16);
  reponse[5] = (uint8_t) ((aileron & 0x0000FF00) >> 8);
  reponse[4] = (uint8_t) (aileron  & 0x000000FF);

  reponse[11] = (uint8_t) ((elevator & 0xFF000000) >> 24);
  reponse[10] = (uint8_t) ((elevator & 0x00FF0000) >> 16);
  reponse[9]  = (uint8_t) ((elevator & 0x0000FF00) >> 8);
  reponse[8]  = (uint8_t) (elevator  & 0x000000FF);

  reponse[15] = (uint8_t) ((throttle & 0xFF000000) >> 24);
  reponse[14] = (uint8_t) ((throttle & 0x00FF0000) >> 16);
  reponse[13] = (uint8_t) ((throttle & 0x0000FF00) >> 8);
  reponse[12] = (uint8_t) (throttle  & 0x000000FF);

  return;
}




void setup() {

  Serial.begin(115200);
  
  //Serial.println("Coming Online");

}


void loop() {
  
  
  //incoming data processing loop
  char magicNumber[3] = {'+', '+', '+'};

  int dataVectorSize = 44;
  int commandVectorSize = 17; //4 commands plus bools
  int controlVectorSize = 16;

  uint8_t rawDataBuffer[64];

  if(Serial.find(magicNumber, 3)){
    //delay(1);

    //Serial.println("got magic number");

    if(Serial.peek() == 'e'){ //for enviroment input
      Serial.read(); //drop that header byte

      //Serial.println("Enviroment");
      Serial.readBytes(rawDataBuffer, dataVectorSize);
      rawDataToEnviroment(rawDataBuffer);

      //does the compute as soon as it recieves the data
      computeControl(controlDeflections);
    }

    if(Serial.peek() == 'c'){ //request control vector
      //control data response
      Serial.read(); //drop that header byte

      //Serial.println("Control");
      uint8_t controlResponse[16] = {};
      controlResponseBytes(controlResponse);
      Serial.write(controlResponse, 16);
    }
    if(Serial.peek() == 'i'){ //command input vector
      //control data response
      Serial.read(); //drop that header byte
      //Serial.println("Input");
      Serial.readBytes(rawDataBuffer, commandVectorSize);
      rawDataToCommands(rawDataBuffer);

      //does the compute on update
      computeControl(controlDeflections);
    }
    if(Serial.peek() == 't'){ //test
      //control data response
      Serial.read(); //drop that header byte

      //Serial.println("Control");
      uint8_t controlResponse[16] = {0, 1, 2, 3, 4, 5};
      Serial.write(controlResponse, 6);
    }
    if(Serial.peek() == 's'){ //state dump
      //control data response
      Serial.read(); //drop that header byte

      Serial.print(" Alt ");
      Serial.print(altitude_msl);
      Serial.print(" EAS ");
      Serial.print(equivalent_airspeed);
      Serial.print(" AOA ");
      Serial.print(angle_of_attack);
      Serial.print(" AOS ");
      Serial.print(angle_of_sideslip);
      Serial.print(" ROL ");
      Serial.print(euler_angle_roll);
      Serial.print(" PCH ");
      Serial.print(euler_angle_pitch);
      Serial.print(" YAW ");
      Serial.print(euler_angle_yaw);
      Serial.print(" RRR ");
      Serial.print(body_angular_rate_roll);
      Serial.print(" RRP ");
      Serial.print(body_angular_rate_pitch);
      Serial.print(" RRY ");
      Serial.print(body_angular_rate_yaw);

      //Serial.println("Control");
      Serial.print(" EASC ");
      Serial.print(equivalent_airspeed_command);
      Serial.print(" ALTC ");
      Serial.print(altitude_msl_command);
      Serial.print(" LATDC ");
      Serial.print(lateral_deviation_error);
      Serial.print(" CRSC ");
      Serial.print(true_base_course_command);
      Serial.print(" STBC ");
      Serial.print(stability_augmentation_on_disc);
      Serial.print(" APOC ");
      Serial.println(autopilot_on_disc);
    }

    Serial.flush();
  }
  
}
