import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import App from './App';

describe('App', () => {
  it('includes version when uploading model in fetchSlice', async () => {
    const mockFetch = vi.fn()
      // Response for /design/review
      .mockResolvedValueOnce(new Response(
        JSON.stringify({ version: 1, spec: [{ id: 'box', modifiers: {} }] }),
        { status: 200 }
      ))
      // Response for POST /models
      .mockResolvedValueOnce(new Response(
        JSON.stringify({ id: 'model-1' }),
        { status: 200 }
      ))
      // Response for GET /models/model-1/slices?layer=0
      .mockResolvedValueOnce(new Response(
        JSON.stringify({ seed_points: [] }),
        { status: 200 }
      ));

    (global as any).fetch = mockFetch;

    const { getByPlaceholderText, getByRole } = render(<App />);
    fireEvent.change(getByPlaceholderText('What do you want to design?'), {
      target: { value: 'box' }
    });
    fireEvent.click(getByRole('button', { name: /generate/i }));

    await waitFor(() => expect(mockFetch).toHaveBeenCalledTimes(3));
    const body = JSON.parse(mockFetch.mock.calls[1][1].body as string);
    expect(body.version).toBe(1);
  });
});

