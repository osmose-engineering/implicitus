import { spawn } from 'child_process';
import path from 'path';

const CARGO = process.env.CARGO || 'cargo';
const cwd = path.join(__dirname, '..', '..', 'core_engine');

const child = spawn(CARGO, ['run', '--bin', 'slicer_server'], {
  cwd,
  stdio: 'inherit'
});

process.on('SIGTERM', () => child.kill());
process.on('SIGINT', () => child.kill());
