import { Component, OnInit, AfterViewInit, Input, ElementRef, Renderer2 } from '@angular/core';
import { Chart } from 'chart.js';

@Component({
  selector: 'app-rainfall-graph',
  standalone: true,
  templateUrl: './rainfall-graph.component.html',
  styleUrls: ['./rainfall-graph.component.css']
})
export class RainfallGraphComponent implements OnInit, AfterViewInit {
  @Input() predictedRainfall: number[];
  //@Input() timeLabels: string[];

  private chart: any;

  constructor(private el: ElementRef, private renderer: Renderer2) {}

  ngOnInit(): void {
    // Any initialization logic that doesn't require the DOM
  }

  ngAfterViewInit(): void {
    this.createChart();
  }

  createChart(): void {
    const ctx = this.el.nativeElement.querySelector('canvas').getContext('2d');
    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: this.timeLabels, // Use the timeLabels input
        datasets: [{
          label: 'Predicted Rainfall',
          data: this.predictedRainfall,
          borderColor: 'rgba(75, 192, 192, 1)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: {
            beginAtZero: true
          }
        }
      }
    });
  }
}