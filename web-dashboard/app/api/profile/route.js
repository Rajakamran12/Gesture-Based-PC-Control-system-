import { NextResponse } from 'next/server';
import { getSupabaseAdmin } from '../../../lib/supabaseServer';

/**
 * Validates the Bearer token from the Authorization header and returns the user.
 */
async function getAuthUser(request) {
  const supabaseAdmin = getSupabaseAdmin();
  const authHeader = request.headers.get('authorization');
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return { user: null, error: 'Missing or invalid Authorization header' };
  }
  const token = authHeader.slice(7);
  const { data: { user }, error } = await supabaseAdmin.auth.getUser(token);
  if (error || !user) {
    return { user: null, error: 'Invalid or expired token' };
  }
  return { user, error: null };
}

/**
 * GET /api/profile
 * Returns the authenticated user's profile.
 */
export async function GET(request) {
  const { user, error: authError } = await getAuthUser(request);
  if (authError) {
    return NextResponse.json({ error: authError }, { status: 401 });
  }

  const { data, error } = await getSupabaseAdmin()
    .from('profiles')
    .select('id, email, name, created_at, updated_at')
    .eq('id', user.id)
    .single();

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json(data);
}

/**
 * POST /api/profile
 * Creates or updates the authenticated user's profile.
 * Body: { name?: string }
 */
export async function POST(request) {
  const { user, error: authError } = await getAuthUser(request);
  if (authError) {
    return NextResponse.json({ error: authError }, { status: 401 });
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  const { name } = body;

  const { data, error } = await getSupabaseAdmin()
    .from('profiles')
    .upsert(
      {
        id: user.id,
        email: user.email,
        name: name ?? null,
        updated_at: new Date().toISOString(),
      },
      { onConflict: 'id' }
    )
    .select('id, email, name, created_at, updated_at')
    .single();

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json(data);
}
