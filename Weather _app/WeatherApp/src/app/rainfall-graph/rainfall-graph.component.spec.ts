import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RainfallGraphComponent } from './rainfall-graph.component';

describe('RainfallGraphComponent', () => {
  let component: RainfallGraphComponent;
  let fixture: ComponentFixture<RainfallGraphComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RainfallGraphComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(RainfallGraphComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
