import { Injectable, Inject, PLATFORM_ID } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, catchError, map, of ,timer,switchMap } from 'rxjs';
import { isPlatformBrowser } from '@angular/common';
import { environment } from '../../environments/environment.development';

interface WeatherData {
  direction: string;
  windspeed: number;
  temperature: number;
  uvRadiation: number;
  cloudCover: number;
  prediction: number[];
  time: string[];
  min_temp:number;
  max_temp:number;
}


@Injectable({
  providedIn: 'root',
})
export class WeatherService {
  private apiUrl = environment.apiUrl; // Ensure `apiUrl` is set

  constructor(private http: HttpClient, @Inject(PLATFORM_ID) private platformId: Object) {}

  getWeatherData(): Observable<WeatherData | null> {
    return this.http.get<WeatherData>(this.apiUrl).pipe(
      map((data: any) => {
        const weatherData: WeatherData = {
          prediction: data.prediction,
          direction: data.direction,
          windspeed: data.windspeed,
          cloudCover: data.cloudCover,
          temperature: data.temperature,
          uvRadiation: data.uvRadiation,
          time: data.time,
          min_temp:data.min_temp,
          max_temp:data.max_temp
        };
        //console.log('Weather Data:', weatherData);
        this.storeWeatherData(weatherData); // Store the data in localStorage
        return weatherData;
      }),
      catchError(error => {
        console.error('Error fetching weather data:', error);
        return of(this.getStoredWeatherData()); // updated to return last data storeed
      })
    );
  }

  storeWeatherData(data: WeatherData) {
    if (isPlatformBrowser(this.platformId)) { // Check if running in a browser
      localStorage.setItem('weatherData', JSON.stringify(data));
    }
  }

  getStoredWeatherData(): WeatherData | null {
    if (isPlatformBrowser(this.platformId)) { // Check if running in a browser
      const data = localStorage.getItem('weatherData');
      return data ? JSON.parse(data) : null;
    }
    return null; // Return null if not in browser
  }
}
