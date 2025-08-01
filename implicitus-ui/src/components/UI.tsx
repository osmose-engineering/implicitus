import React from 'react';

export interface CheckboxProps {
  checked: boolean;
  label: string;
  onChange: (value: boolean) => void;
  style?: React.CSSProperties;
}

export const Checkbox: React.FC<CheckboxProps> = ({ checked, label, onChange, style }) => (
  <label style={style}>
    <input
      type="checkbox"
      checked={checked}
      onChange={e => onChange(e.target.checked)}
      style={{ marginRight: '0.5em' }}
    />
    {label}
  </label>
);

export interface SliderProps {
  value: number;
  min: number;
  max: number;
  step?: number;
  label: string;
  onChange: (value: number) => void;
  style?: React.CSSProperties;
}

export const Slider: React.FC<SliderProps> = ({ value, min, max, step, label, onChange, style }) => (
  <label style={{ display: 'inline-block', ...style }}>
    <div style={{ marginBottom: '0.25em' }}>{label}</div>
    <input
      type="range"
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={e => onChange(parseFloat(e.target.value))}
    />
  </label>
);