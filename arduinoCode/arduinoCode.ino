#include "units.h"
#include "wmm.h"

void setup() {
  // Start the serial communication
  Serial.begin(9600);
  
  // Give some time for the serial monitor to open
  delay(2000);

  // Specify the parameters for the magnetic field calculation
  float altitude = 1000.0;       // Altitude in meters above WGS84 ellipsoid
  float latitude = 47.6062;      // Latitude in degrees (example: Seattle, WA)
  float longitude = -122.3321;   // Longitude in degrees (example: Seattle, WA)
  float decimal_year = 2024.0;   // Current year as a decimal
  bfs::WmmModel model = bfs::WMM2020; // Use WMM2020 model

  // Calculate the magnetic field data
  bfs::WmmData data = bfs::wrldmagm(altitude, latitude, longitude, decimal_year, model);

  // Print the magnetic field components and other data
  Serial.println("Magnetic Field Data:");
  Serial.print("North Component (nT): ");
  Serial.println(data.mag_field_nt[0]);
  
  Serial.print("East Component (nT): ");
  Serial.println(data.mag_field_nt[1]);
  
  Serial.print("Vertical Component (nT): ");
  Serial.println(data.mag_field_nt[2]);
  
  Serial.print("Horizontal Intensity (nT): ");
  Serial.println(data.horz_intensity_nt);
  
  Serial.print("Declination (degrees): ");
  Serial.println(data.declination_deg);
  
  Serial.print("Inclination (degrees): ");
  Serial.println(data.inclination_deg);
  
  Serial.print("Total Intensity (nT): ");
  Serial.println(data.total_intensity_nt);
}

void loop() {
  // The program only needs to run once in the setup, so no code is needed here
}