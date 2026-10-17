declare module 'lucide-react' {
  import type { ComponentType, SVGProps } from 'react'

  type IconProps = SVGProps<SVGSVGElement> & {
    size?: number | string
    strokeWidth?: number | string
  }

  export const Activity: ComponentType<IconProps>
  export const ArrowDown: ComponentType<IconProps>
  export const ArrowUp: ComponentType<IconProps>
  export const Check: ComponentType<IconProps>
  export const ChevronDown: ComponentType<IconProps>
  export const FileText: ComponentType<IconProps>
  export const FlaskConical: ComponentType<IconProps>
  export const FolderOpen: ComponentType<IconProps>
  export const LoaderCircle: ComponentType<IconProps>
  export const Menu: ComponentType<IconProps>
  export const MessageSquareText: ComponentType<IconProps>
  export const Paperclip: ComponentType<IconProps>
  export const Plus: ComponentType<IconProps>
  export const Search: ComponentType<IconProps>
  export const ShieldCheck: ComponentType<IconProps>
  export const Sparkles: ComponentType<IconProps>
  export const X: ComponentType<IconProps>
}
