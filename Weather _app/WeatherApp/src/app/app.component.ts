import { Component, OnInit } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { CommonModule } from '@angular/common';
import { WeatherService } from './Service/weather.service'; // Adjust the path if necessary
import { HttpClientModule } from '@angular/common/http';
import { CanvasJSAngularChartsModule } from '@canvasjs/angular-charts';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, CommonModule, HttpClientModule, CanvasJSAngularChartsModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'WeatherApp';
  time: string[]=[];
  rainfall: number []=[];
  direction: string | undefined;
  windSpeed: number | undefined;
  temperature: number | undefined;
  uvRadiation: number | undefined;
  cloudCover: number | undefined;
  min_temp: number | undefined;
  max_temp: number | undefined;
  predictedRainfall: number[] = [0, 2, 5, 1, 0, 3]; // Example predicted rainfall data
  timeLabels: string[] = ['12:00', '1:00', '2:00', '3:00', '4:00', '5:00']; // Example time labels 
  constructor(private weatherService: WeatherService) {}

  ngOnInit() {
    this.getRainfallData(); // Fetch rainfall data when the component initializes
  }

 getRainfallData() {
  this.weatherService.getWeatherData().subscribe(
    (response) => {
      this.time = response?.time|| [];
      this.direction = response?.direction;
      this.windSpeed = response?.windspeed;
      this.temperature = response?.temperature;
      this.uvRadiation = response?.uvRadiation;
      this.cloudCover = response?.cloudCover;
      this.rainfall = response?.prediction||[];  // Store data for display
      this.min_temp=response?.min_temp;
      this.max_temp=response?.max_temp;
     
    },
    
    (error) => {
      console.error('Error fetching data', error);
    }
  );
}


  updateChart() {
    // Logic to update chart if required, based on fetched data
  }

  get chartOptions() {
    return {
      theme: 'dark2',
      width: 750, // Set a specific width
      height: 310, // Set a specific height
      animationEnabled: true,
      title: {
        text: "Hourly Rainfall",
        fontFamily: "Arial",
        fontColor: "#FCFAEE",
        fontSize: 26
      },
      axisX: {
        title: "Time",
        includeZero: false,
        labelFontSize: 14
      },
      axisY: {
        title: "Precipitation (mm)",
        labelFontSize: 14
      },
      toolTip: { shared: false },
      legend: {
        cursor: "pointer",
        itemclick: function(e: any) {
          e.dataSeries.visible = typeof(e.dataSeries.visible) === "undefined" || e.dataSeries.visible ? false : true;
          e.chart.render();
        }
      },
      data: [
        {
          type: "spline",
          showInLegend: true,
          name: "Rainfall",
          dataPoints: this.time.map((label, index) => ({
            label: label,
            y: this.rainfall[index]
          }))
        }
      ],
      responsive: true,
      maintainAspectRatio: false,
    };
  }
}
