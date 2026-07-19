import { useMemo } from 'react'
import * as THREE from 'three'
import { Canvas } from '@react-three/fiber'
import { ContactShadows, OrbitControls } from '@react-three/drei'
import type { Color } from '../types'

// Melissa & Doug-style block set: door = arch, wall = rectangle, roof = triangle,
// stacked bottom to top exactly as the robot builds them.
const PLACEHOLDER = '#d4d4d8'
const BLOCK_WIDTH = 2.2
const BLOCK_DEPTH = 1.3
const DOOR_HEIGHT = 1.3
const WALL_HEIGHT = 1.1
const ROOF_HEIGHT = 1.1

// Exact hex values from the physical Melissa & Doug block set.
const COLOR_SWATCH: Record<Color, string> = {
  red: '#CF2732',
  blue: '#0068A3',
  yellow: '#FFE400',
  green: '#00892A',
}

export interface HouseSelection {
  door: Color | null
  wall: Color | null
  roof: Color | null
}

function centeredExtrude(shape: THREE.Shape, height: number, depth: number): THREE.ExtrudeGeometry {
  const geometry = new THREE.ExtrudeGeometry(shape, {
    depth,
    bevelEnabled: false,
    curveSegments: 32,
  })
  geometry.translate(0, -height / 2, -depth / 2)
  return geometry
}

function archShape(): THREE.Shape {
  const shape = new THREE.Shape()
  shape.moveTo(-BLOCK_WIDTH / 2, 0)
  shape.lineTo(BLOCK_WIDTH / 2, 0)
  shape.lineTo(BLOCK_WIDTH / 2, DOOR_HEIGHT)
  shape.lineTo(-BLOCK_WIDTH / 2, DOOR_HEIGHT)
  shape.closePath()

  // Doorway cutout: straight legs topped with a semicircular arch.
  const openingRadius = (BLOCK_WIDTH * 0.45) / 2
  const legHeight = DOOR_HEIGHT * 0.35
  const hole = new THREE.Path()
  hole.moveTo(-openingRadius, 0)
  hole.lineTo(-openingRadius, legHeight)
  hole.absarc(0, legHeight, openingRadius, Math.PI, 0, true)
  hole.lineTo(openingRadius, 0)
  hole.closePath()
  shape.holes.push(hole)

  return shape
}

function roofShape(): THREE.Shape {
  // Slightly wider than the wall/door so the roof overhangs, like a real pitched roof.
  const overhang = 0.15
  const shape = new THREE.Shape()
  shape.moveTo(-BLOCK_WIDTH / 2 - overhang, 0)
  shape.lineTo(BLOCK_WIDTH / 2 + overhang, 0)
  shape.lineTo(0, ROOF_HEIGHT)
  shape.closePath()
  return shape
}

function ArchBlock({ color, y }: { color: string; y: number }) {
  const shape = useMemo(() => archShape(), [])
  const geometry = useMemo(() => centeredExtrude(shape, DOOR_HEIGHT, BLOCK_DEPTH), [shape])
  return (
    <mesh geometry={geometry} position={[0, y, 0]} castShadow receiveShadow>
      <meshStandardMaterial color={color} roughness={0.5} />
    </mesh>
  )
}

function WallBlock({ color, y }: { color: string; y: number }) {
  return (
    <mesh position={[0, y, 0]} castShadow receiveShadow>
      <boxGeometry args={[BLOCK_WIDTH, WALL_HEIGHT, BLOCK_DEPTH]} />
      <meshStandardMaterial color={color} roughness={0.5} />
    </mesh>
  )
}

function RoofBlock({ color, y }: { color: string; y: number }) {
  const shape = useMemo(() => roofShape(), [])
  const geometry = useMemo(() => centeredExtrude(shape, ROOF_HEIGHT, BLOCK_DEPTH + 0.3), [shape])
  return (
    <mesh geometry={geometry} position={[0, y, 0]} castShadow receiveShadow>
      <meshStandardMaterial color={color} roughness={0.5} />
    </mesh>
  )
}

export function HousePreview3D({ selection }: { selection: HouseSelection }) {
  const doorColor = selection.door ? COLOR_SWATCH[selection.door] : PLACEHOLDER
  const wallColor = selection.wall ? COLOR_SWATCH[selection.wall] : PLACEHOLDER
  const roofColor = selection.roof ? COLOR_SWATCH[selection.roof] : PLACEHOLDER

  const doorY = DOOR_HEIGHT / 2
  const wallY = DOOR_HEIGHT + WALL_HEIGHT / 2
  const roofY = DOOR_HEIGHT + WALL_HEIGHT + ROOF_HEIGHT / 2
  const stackHeight = DOOR_HEIGHT + WALL_HEIGHT + ROOF_HEIGHT

  return (
    <div className="house-preview-3d">
      <Canvas shadows camera={{ position: [3.6, 2.6, 4.4], fov: 40 }}>
        <ambientLight intensity={0.7} />
        <directionalLight position={[4, 6, 3]} intensity={1.4} castShadow shadow-mapSize={[1024, 1024]} />
        <group position={[0, -stackHeight / 2, 0]}>
          <ArchBlock color={doorColor} y={doorY} />
          <WallBlock color={wallColor} y={wallY} />
          <RoofBlock color={roofColor} y={roofY} />
          <ContactShadows position={[0, 0, 0]} opacity={0.45} scale={7} blur={2.2} far={4} />
        </group>
        <OrbitControls enablePan={false} minDistance={3} maxDistance={9} />
      </Canvas>
    </div>
  )
}
