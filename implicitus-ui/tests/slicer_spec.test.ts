import { describe, it, expect, vi } from 'vitest';
import { getSpecArray } from '../src/App';

async function run(data: any, slicer: (model: any) => Promise<void> | void) {
  const specArray = getSpecArray(data);
  if (Array.isArray(specArray)) {
    await slicer({ id: 'preview', root: { children: specArray } });
  }
}

describe('slicer invocation', () => {
  it('invokes slicer when response spec is array', async () => {
    const slicer = vi.fn();
    const data = { spec: [{ modifiers: { infill: { seed_points: [] } } }] };
    await run(data, slicer);
    expect(slicer).toHaveBeenCalled();
  });

  it('invokes slicer when response spec is object containing spec array', async () => {
    const slicer = vi.fn();
    const data = { spec: { version: 1, spec: [{ modifiers: { infill: { seed_points: [] } } }] } };
    await run(data, slicer);
    expect(slicer).toHaveBeenCalled();
  });
});
