import { useCallback, useEffect, useRef, useState } from 'react';
import type { RealtimeChannel } from '@supabase/supabase-js';
import { cloudEnabled, getSupabase } from '../lib/supabase';
import type {
  AvatarPresetId,
  GameSceneId,
  LocalPositionEvent,
  RemotePlayerState,
  WorldPoint,
} from './types';

type PresencePayload = {
  clientId: string;
  userId: string;
  displayName: string;
  avatarPreset: AvatarPresetId;
  sceneId: GameSceneId;
  x: number;
  y: number;
};

type RemoteEmote = { clientId: string; emote: 'wave'; at: number };

export function useGamePresence(options: {
  userId: string | null;
  displayName: string;
  avatarPreset: AvatarPresetId;
  sceneId: GameSceneId;
  initialPosition: WorldPoint;
}) {
  const { userId, displayName, avatarPreset, sceneId, initialPosition } = options;
  const clientIdRef = useRef(crypto.randomUUID());
  const channelRef = useRef<RealtimeChannel | null>(null);
  const remotesRef = useRef(new Map<string, RemotePlayerState>());
  const lastSentRef = useRef(0);
  const [remotePlayers, setRemotePlayers] = useState<RemotePlayerState[]>([]);
  const [remoteEmote, setRemoteEmote] = useState<RemoteEmote | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    if (!cloudEnabled || !userId) {
      return;
    }
    const client = getSupabase();
    const clientId = clientIdRef.current;
    const remotes = remotesRef.current;
    const channel = client.channel(`game:scene:${sceneId}`, {
      config: {
        broadcast: { self: false, ack: false },
        presence: { key: clientId },
      },
    });
    channelRef.current = channel;

    const publishRemotes = () => {
      setRemotePlayers([...remotes.values()]);
    };

    channel
      .on('presence', { event: 'sync' }, () => {
        const presence = channel.presenceState<PresencePayload>();
        const active = new Set<string>();
        for (const entries of Object.values(presence)) {
          for (const entry of entries) {
            if (entry.clientId === clientId) continue;
            active.add(entry.clientId);
            const previous = remotes.get(entry.clientId);
            remotes.set(entry.clientId, {
              clientId: entry.clientId,
              userId: entry.userId,
              displayName: entry.displayName,
              avatarPreset: entry.avatarPreset,
              sceneId: entry.sceneId,
              x: previous?.x ?? entry.x,
              y: previous?.y ?? entry.y,
              direction: previous?.direction ?? 'right',
              updatedAt: previous?.updatedAt ?? Date.now(),
            });
          }
        }
        for (const remoteId of remotes.keys()) {
          if (!active.has(remoteId)) remotes.delete(remoteId);
        }
        publishRemotes();
      })
      .on('broadcast', { event: 'position' }, ({ payload }) => {
        const state = payload as RemotePlayerState;
        if (!state?.clientId || state.clientId === clientId || state.sceneId !== sceneId) return;
        remotes.set(state.clientId, { ...state, updatedAt: Date.now() });
        publishRemotes();
      })
      .on('broadcast', { event: 'emote' }, ({ payload }) => {
        const event = payload as RemoteEmote;
        if (event?.clientId && event.clientId !== clientId && event.emote === 'wave') {
          setRemoteEmote({ ...event });
        }
      })
      .subscribe(async (status) => {
        const ready = status === 'SUBSCRIBED';
        setConnected(ready);
        if (!ready) return;
        const payload: PresencePayload = {
          clientId,
          userId,
          displayName,
          avatarPreset,
          sceneId,
          x: initialPosition.x,
          y: initialPosition.y,
        };
        await channel.track(payload);
      });

    return () => {
      channelRef.current = null;
      remotes.clear();
      setRemotePlayers([]);
      setConnected(false);
      void client.removeChannel(channel);
    };
  }, [avatarPreset, displayName, initialPosition.x, initialPosition.y, sceneId, userId]);

  const broadcastPosition = useCallback(
    (position: LocalPositionEvent) => {
      const channel = channelRef.current;
      if (!channel || !userId || position.sceneId !== sceneId) return;
      const now = Date.now();
      if (now - lastSentRef.current < 110) return;
      lastSentRef.current = now;
      const state: RemotePlayerState = {
        clientId: clientIdRef.current,
        userId,
        displayName,
        avatarPreset,
        sceneId,
        x: position.x,
        y: position.y,
        direction: position.direction,
        updatedAt: now,
      };
      void channel.send({ type: 'broadcast', event: 'position', payload: state });
    },
    [avatarPreset, displayName, sceneId, userId],
  );

  const broadcastEmote = useCallback((emote: 'wave') => {
    const channel = channelRef.current;
    if (!channel) return;
    void channel.send({
      type: 'broadcast',
      event: 'emote',
      payload: { clientId: clientIdRef.current, emote, at: Date.now() },
    });
  }, []);

  return { connected, remotePlayers, remoteEmote, broadcastPosition, broadcastEmote };
}
