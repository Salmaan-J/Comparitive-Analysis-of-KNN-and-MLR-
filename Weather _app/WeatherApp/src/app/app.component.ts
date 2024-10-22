import { Component, OnInit } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { CommonModule } from '@angular/common';
import { WeatherService } from './Service/weather.service'; // Adjust the path if necessary
import { HttpClientModule } from '@angular/common/http';
import { RainfallGraphComponent } from './rainfall-graph/rainfall-graph.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, CommonModule, HttpClientModule, RainfallGraphComponent],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'WeatherApp';
  rainfall: number | undefined;
  predictedRainfall: number[] = [0, 2, 5, 1, 0, 3]; // Example predicted rainfall data
  //timeLabels: string[] = ['12:00', '1:00', '2:00', '3:00', '4:00', '5:00']; // Example time labels 

  constructor(private weatherService: WeatherService) {}

  ngOnInit() {
    this.getRainfallData(); // Fetch rainfall data when the component initializes
  }

  getRainfallData() {
    this.weatherService.getWeatherData().subscribe(
      (data) => {
        this.rainfall = data.rainfall; // Store the rainfall data
        console.log(this.rainfall); // Testing to view data received from API
      },
      (error) => {
        console.error('Error fetching rainfall data', error); // Handle error
      }
    );
  }
}