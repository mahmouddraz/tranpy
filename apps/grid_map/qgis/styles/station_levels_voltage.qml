<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0-Prizren" styleCategories="Symbology|Labeling">
  <renderer-v2 type="RuleRenderer" symbollevels="0" enableorderby="0" forceraster="0" referencescale="-1">
    <rules key="{root}">
      <rule key="{r0}" label="Busbar" symbol="0" filter="NOT (&quot;part&quot; = 'stub')" scalemaxdenom="70000" scalemindenom="1" />
      <rule key="{r1}" label="380 kV link from the edge port to the busbar" symbol="1" filter="&quot;part&quot; = 'stub' AND &quot;voltage_kv&quot; &gt;= 300" scalemaxdenom="70000" scalemindenom="1" />
      <rule key="{r2}" label="220 kV link from the edge port to the busbar" symbol="2" filter="&quot;part&quot; = 'stub' AND &quot;voltage_kv&quot; &lt; 300" scalemaxdenom="70000" scalemindenom="1" />
    </rules>
    <symbols>
      <symbol type="line" name="0" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="line_color" type="QString" value="38,50,56,255" />
            <Option name="line_width" type="QString" value="1.6" />
            <Option name="line_width_unit" type="QString" value="MM" />
            <Option name="line_style" type="QString" value="solid" />
            <Option name="capstyle" type="QString" value="round" />
            <Option name="joinstyle" type="QString" value="round" />
            <Option name="use_custom_dash" type="QString" value="0" />
            <Option name="draw_inside_polygon" type="QString" value="0" />
            <Option name="offset" type="QString" value="0" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map" />
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="line" name="1" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="line_color" type="QString" value="215,25,28,191" />
            <Option name="line_width" type="QString" value="0.2" />
            <Option name="line_width_unit" type="QString" value="MM" />
            <Option name="line_style" type="QString" value="solid" />
            <Option name="capstyle" type="QString" value="round" />
            <Option name="joinstyle" type="QString" value="round" />
            <Option name="use_custom_dash" type="QString" value="0" />
            <Option name="draw_inside_polygon" type="QString" value="0" />
            <Option name="offset" type="QString" value="0" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map" />
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="line" name="2" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="line_color" type="QString" value="26,150,65,191" />
            <Option name="line_width" type="QString" value="0.2" />
            <Option name="line_width_unit" type="QString" value="MM" />
            <Option name="line_style" type="QString" value="solid" />
            <Option name="capstyle" type="QString" value="round" />
            <Option name="joinstyle" type="QString" value="round" />
            <Option name="use_custom_dash" type="QString" value="0" />
            <Option name="draw_inside_polygon" type="QString" value="0" />
            <Option name="offset" type="QString" value="0" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map" />
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <layerGeometryType>1</layerGeometryType>
</qgis>
