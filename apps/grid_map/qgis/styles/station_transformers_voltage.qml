<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0-Prizren" styleCategories="Symbology|Labeling">
  <renderer-v2 type="RuleRenderer" symbollevels="0" enableorderby="0" forceraster="0" referencescale="-1">
    <rules key="{root}">
      <rule key="{r0}" label="Phase-shifting transformer" symbol="0" filter="&quot;phase_shifter&quot; = 1" scalemaxdenom="70000" scalemindenom="1" />
      <rule key="{r1}" label="Transformer (HV circle left, LV circle right; colour = voltage)" symbol="1" filter="&quot;phase_shifter&quot; = 0" scalemaxdenom="70000" scalemindenom="1" />
    </rules>
    <symbols>
      <symbol type="marker" name="0" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="name" type="QString" value="diamond" />
            <Option name="color" type="QString" value="106,61,154,255" />
            <Option name="outline_color" type="QString" value="255,255,255,255" />
            <Option name="outline_style" type="QString" value="solid" />
            <Option name="outline_width" type="QString" value="0.3" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="size" type="QString" value="2.6" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="miter" />
            <Option name="scale_method" type="QString" value="diameter" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
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
      <symbol type="marker" name="1" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="name" type="QString" value="circle" />
            <Option name="color" type="QString" value="255,255,255,153" />
            <Option name="outline_color" type="QString" value="69,90,100,255" />
            <Option name="outline_style" type="QString" value="solid" />
            <Option name="outline_width" type="QString" value="0.4" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="size" type="QString" value="1.9" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="-0.5,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="miter" />
            <Option name="scale_method" type="QString" value="diameter" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="outlineColor" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="CASE WHEN &quot;kv_hv&quot; &gt;= 300 THEN '215,25,28,255' WHEN &quot;kv_hv&quot; &gt;= 200 THEN '26,150,65,255' ELSE '85,85,85,255' END" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="name" type="QString" value="circle" />
            <Option name="color" type="QString" value="255,255,255,153" />
            <Option name="outline_color" type="QString" value="69,90,100,255" />
            <Option name="outline_style" type="QString" value="solid" />
            <Option name="outline_width" type="QString" value="0.4" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="size" type="QString" value="1.9" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0.5,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="miter" />
            <Option name="scale_method" type="QString" value="diameter" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="outlineColor" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="CASE WHEN &quot;kv_lv&quot; &gt;= 300 THEN '215,25,28,255' WHEN &quot;kv_lv&quot; &gt;= 200 THEN '26,150,65,255' ELSE '85,85,85,255' END" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <layerGeometryType>0</layerGeometryType>
</qgis>
