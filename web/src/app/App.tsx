interface AppProps {
  detailPlaceholder?: boolean
}

export function App({ detailPlaceholder = false }: AppProps) {
  return (
    <section aria-labelledby="planner-foundation-title" className="route-state">
      <p className="eyebrow">Workbench foundation</p>
      <h2 id="planner-foundation-title">
        {detailPlaceholder ? 'Configuration detail' : 'Planner workbench'}
      </h2>
      <p>Demonstration data interaction is being connected to this product shell.</p>
    </section>
  )
}
